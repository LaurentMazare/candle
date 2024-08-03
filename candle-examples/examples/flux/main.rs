#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use candle_transformers::models::{clip, flux, t5};

use anyhow::{Error as E, Result};
use candle::{Module, Tensor};
use candle_nn::VarBuilder;
use clap::Parser;
use tokenizers::Tokenizer;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// The prompt to be used for image generation.
    #[arg(
        long,
        default_value = "A very realistic photo of a rusty robot walking on a sandy beach"
    )]
    prompt: String,

    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The height in pixels of the generated image.
    #[arg(long)]
    height: Option<usize>,

    /// The width in pixels of the generated image.
    #[arg(long)]
    width: Option<usize>,
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
    let device = candle_examples::device(cpu)?;
    let dtype = device.bf16_default_to_f32();
    let t5_emb = {
        let repo = api.repo(hf_hub::Repo::with_revision(
            "google/t5-v1_1-xxl".to_string(),
            hf_hub::RepoType::Model,
            "refs/pr/2".to_string(),
        ));
        let model_file = repo.get("model.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let config_filename = repo.get("config.json")?;
        let config = std::fs::read_to_string(config_filename)?;
        let config: t5::Config = serde_json::from_str(&config)?;
        let mut model = t5::T5EncoderModel::load(vb, &config)?;
        let tokenizer_filename = api
            .model("lmz/mt5-tokenizers".to_string())
            .get("t5-v1_1-xxl.tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let input_token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        model.forward(&input_token_ids)?
    };
    println!("T5\n{t5_emb}");
    let clip_emb = {
        let repo = api.repo(hf_hub::Repo::model(
            "openai/clip-vit-large-patch14".to_string(),
        ));
        let model_file = repo.get("model.safetensors")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        // https://huggingface.co/openai/clip-vit-large-patch14/blob/main/config.json
        let config = clip::text_model::ClipTextConfig {
            vocab_size: 49408,
            projection_dim: 768,
            activation: clip::text_model::Activation::QuickGelu,
            intermediate_size: 3072,
            embed_dim: 768,
            max_position_embeddings: 77,
            pad_with: None,
            num_hidden_layers: 12,
            num_attention_heads: 12,
        };
        let config = clip::EncoderConfig::Text(config);
        let model = clip::text_model::ClipEncoder::new(vb, &config)?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt.as_str(), true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        let input_token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        model.forward(&input_token_ids, None)?
    };
    println!("CLIP\n{clip_emb}");
    let repo = api.repo(hf_hub::Repo::model(
        "black-forest-labs/FLUX.1-schnell".to_string(),
    ));
    let img = {
        let model_file = repo.get("flux1-schnell.sft")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let cfg = flux::model::Config::schnell();
        let model = flux::model::Flux::new(&cfg, vb)?;

        let img = flux::sampling::get_noise(1, height, width, &device)?;
        let state = flux::sampling::State::new(&t5_emb, &clip_emb, &img)?;
        let timesteps = flux::sampling::get_schedule(4, None); // no shift for flux-schnell
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
    };
    println!("latent img\n{img}");
    let img = {
        let model_file = repo.get("ae.sft")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_file], dtype, &device)? };
        let cfg = flux::autoencoder::Config::schnell();
        let model = flux::autoencoder::AutoEncoder::new(&cfg, vb)?;
        model.forward(&img)?
    };
    println!("img\n{img}");
    candle_examples::save_image(&img, "out.jpg")?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    run(args)
}
