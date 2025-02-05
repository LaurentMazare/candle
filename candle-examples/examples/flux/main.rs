#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::{clip, flux, t5};

use anyhow::{Error as E, Result};
use candle::{IndexOp, Module, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(long, default_value = "A rusty robot walking on a beach")]
    prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Use the quantized model.
    #[arg(long)]
    quantized: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,

    #[arg(long)]
    decode_only: Option<String>,

    #[arg(long, value_enum, default_value = "schnell")]
    model: Model,

    /// Use the slower kernels.
    #[arg(long)]
    use_dmmv: bool,

    /// The seed to use when generating random samples.
    #[arg(long)]
    seed: Option<u64>,
}

#[derive(Debug, Clone, Copy, clap::ValueEnum, PartialEq, Eq)]
enum Model {
    Schnell,
    Dev,
}

fn run(args: Args) -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let Args {
        prompt,
        cpu,
        height,
        width,
        tracing,
        decode_only,
        model,
        quantized,
        ..
    } = args;
    let width = width.unwrap_or(1360);
    let height = height.unwrap_or(768);

    let _guard = if tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let api = hf_hub::api::sync::Api::new()?;
    let bf_repo = {
        let name = match model {
            Model::Dev => "black-forest-labs/FLUX.1-dev",
            Model::Schnell => "black-forest-labs/FLUX.1-schnell",
        };
        api.repo(hf_hub::Repo::model(name.to_string()))
    };
    let device = candle_examples::device(cpu)?;
    if let Some(seed) = args.seed {
        device.set_seed(seed)?;
    }
    let dtype = device.bf16_default_to_f32();
    let img = match decode_only {
        None => {
            let t5_emb = {
                let repo = api.repo(hf_hub::Repo::with_revision(
                    "google/t5-v1_1-xxl".to_string(),
                    hf_hub::RepoType::Model,
                    "refs/pr/2".to_string(),
                ));
                let model_file = repo.get("model.safetensors")?;
                let vb =
                    unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
                let config_filename = repo.get("config.json")?;
                let config = std::fs::read_to_string(config_filename)?;
                let config: t5::Config = serde_json::from_str(&config)?;
                let mut model = t5::T5EncoderModel::load(vb, &config)?;
                let tokenizer_filename = api
                    .model("lmz/mt5-tokenizers".to_string())
                    .get("t5-v1_1-xxl.tokenizer.json")?;
                let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
                let mut tokens = tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                tokens.resize(256, 0);
                let input_token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
                println!("{input_token_ids}");
                model.forward(&input_token_ids)?
            };
            println!("T5\n{t5_emb}");
            let clip_emb = {
                let repo = api.repo(hf_hub::Repo::model(
                    "openai/clip-vit-large-patch14".to_string(),
                ));
                let model_file = repo.get("model.safetensors")?;
                let vb =
                    unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
                // https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
                let config = clip::text_model::ClipTextConfig {
                    vocab_size: 49408,
                    projection_dim: 768,
                    hidden_act: clip::text_model::Activation::QuickGelu,
                    intermediate_size: 3072,
                    hidden_size: 768,
                    max_position_embeddings: 77,
                    pad_with: None,
                    num_hidden_layers: 12,
                    num_attention_heads: 12,
                };
                let model =
                    clip::text_model::ClipTextTransformer::new(vb.pp("text_model"), &config)?;
                let tokenizer_filename = repo.get("tokenizer.json")?;
                let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
                let tokens = tokenizer
                    .encode(prompt.as_str(), true)
                    .map_err(E::msg)?
                    .get_ids()
                    .to_vec();
                let input_token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
                println!("{input_token_ids}");
                model.forward(&input_token_ids)?
            };
            println!("CLIP\n{clip_emb}");
            let img = {
                let cfg = match model {
                    Model::Dev => flux::model::Config::dev(),
                    Model::Schnell => flux::model::Config::schnell(),
                };
                let img = flux::sampling::get_noise(1, height, width, &device)?.to_dtype(dtype)?;
                let state = if quantized {
                    flux::sampling::State::new(
                        &t5_emb.to_dtype(candle::DType::F32)?,
                        &clip_emb.to_dtype(candle::DType::F32)?,
                        &img.to_dtype(candle::DType::F32)?,
                    )?
                } else {
                    flux::sampling::State::new(&t5_emb, &clip_emb, &img)?
                };
                let timesteps = match model {
                    Model::Dev => {
                        flux::sampling::get_schedule(50, Some((state.img.dim(1)?, 0.5, 1.15)))
                    }
                    Model::Schnell => flux::sampling::get_schedule(4, None),
                };
                println!("{state:?}");
                println!("{timesteps:?}");
                if quantized {
                    let model_file = match model {
                        Model::Schnell => api
                            .repo(hf_hub::Repo::model("lmz/candle-flux".to_string()))
                            .get("flux1-schnell.gguf")?,
                        Model::Dev => todo!(),
                    };
                    let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                        model_file, &device,
                    )?;

                    let model = flux::quantized_model::Flux::new(&cfg, vb)?;
                    flux::sampling::denoise(
                        &model,
                        &state.img,
                        &state.img_ids,
                        &state.txt,
                        &state.txt_ids,
                        &state.vec,
                        &timesteps,
                        4.,
                    )?
                    .to_dtype(dtype)?
                } else {
                    let model_file = match model {
                        Model::Schnell => bf_repo.get("flux1-schnell.safetensors")?,
                        Model::Dev => bf_repo.get("flux1-dev.safetensors")?,
                    };
                    let vb = unsafe {
                        VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)?
                    };
                    let model = flux::model::Flux::new(&cfg, vb)?;
                    flux::sampling::denoise(
                        &model,
                        &state.img,
                        &state.img_ids,
                        &state.txt,
                        &state.txt_ids,
                        &state.vec,
                        &timesteps,
                        4.,
                    )?
                }
            };
            flux::sampling::unpack(&img, height, width)?
        }
        Some(file) => {
            let mut st = candle::safetensors::load(file, &device)?;
            st.remove("img").unwrap().to_dtype(dtype)?
        }
    };
    println!("latent img\n{img}");

    let img = {
        let model_file = bf_repo.get("ae.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let cfg = match model {
            Model::Dev => flux::autoencoder::Config::dev(),
            Model::Schnell => flux::autoencoder::Config::schnell(),
        };
        let model = flux::autoencoder::AutoEncoder::new(&cfg, vb)?;
        model.decode(&img)?
    };
    println!("img\n{img}");
    let img = ((img.clamp(-1f32, 1f32)? + 1.0)? * 127.5)?.to_dtype(candle::DType::U8)?;
    let filename = match args.seed {
        None => "out.jpg".to_string(),
        Some(s) => format!("out-{s}.jpg"),
    };
    candle_examples::save_image(&img.i(0)?, filename)?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(feature = "cuda")]
    candle::quantized::cuda::set_force_dmmv(args.use_dmmv);
    run(args)
}
