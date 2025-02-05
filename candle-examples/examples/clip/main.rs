#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Error as E;
use clap::Parser;

use candle::{DType, Device, Tensor};
use candle_nn::{ops::softmax, VarBuilder};
use candle_transformers::models::clip::{self, ClipConfig};

use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

#[derive(Parser)]
struct Args {
    /// model to use from local file. If not provided will download from huggingface
    #[arg(long)]
    model: Option<String>,

    #[arg(long)]
    tokenizer: Option<String>,

    /// The images to use
    #[arg(long, use_value_delimiter = true)]
    images: Option<Vec<String>>,

    #[arg(long)]
    cpu: bool,

    /// The sequences to use
    #[arg(long, use_value_delimiter = true)]
    sequences: Option<Vec<String>>,

    /// The model_id to use from huggingface, check out available models:https://huggingface.co/openai
    #[arg(long)]
    model_id: Option<String>,

    /// The revision to use from huggingface
    #[arg(long)]
    revision: Option<String>,

    /// Use the pytorch weights rather than the safetensors ones
    #[arg(long, default_value = "false")]
    use_pth: bool,
}

fn load_image<T: AsRef<std::path::Path>>(path: T, image_size: usize) -> anyhow::Result<Tensor> {
    let img = image::ImageReader::open(path)?.decode()?;
    let (height, width) = (image_size, image_size);
    let img = img.resize_to_fill(
        width as u32,
        height as u32,
        image::imageops::FilterType::Triangle,
    );
    let img = img.to_rgb8();
    let img = img.into_raw();
    let img = Tensor::from_vec(img, (height, width, 3), &Device::Cpu)?
        .permute((2, 0, 1))?
        .to_dtype(DType::F32)?
        .affine(2. / 255., -1.)?;
    Ok(img)
}

fn load_images<T: AsRef<std::path::Path>>(
    paths: &Vec<T>,
    image_size: usize,
) -> anyhow::Result<Tensor> {
    let mut images = vec![];
    for path in paths {
        let tensor = load_image(path, image_size)?;
        images.push(tensor);
    }
    let images = Tensor::stack(&images, 0)?;
    Ok(images)
}

pub fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt::init();

    let device = candle_examples::device(args.cpu)?;
    let default_model = "openai/clip-vit-base-patch16".to_string();
    let default_revision = "refs/pr/4".to_string();
    let (model_id, revision) = match (args.model_id.to_owned(), args.revision.to_owned()) {
        (Some(model_id), Some(revision)) => (model_id, revision),
        (Some(model_id), None) => (model_id, "main".to_string()),
        (None, Some(revision)) => (default_model, revision),
        (None, None) => (default_model, default_revision),
    };

    let repo = Repo::with_revision(model_id, RepoType::Model, revision);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = match args.tokenizer {
            Some(tokenizer) => tokenizer.into(),
            None => api.get("tokenizer.json")?,
        };
        let weights = match args.model {
            Some(model) => model.into(),
            None => {
                if args.use_pth {
                    api.get("pytorch_model.bin")?
                } else {
                    api.get("model.safetensors")?
                }
            }
        };

        (config, tokenizer, weights)
    };

    let config: String = std::fs::read_to_string(config_filename)?;
    let config: ClipConfig = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vec_imgs = match args.images {
        Some(imgs) => imgs,
        None => vec![
            "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg".to_string(),
            "candle-examples/examples/yolo-v8/assets/bike.jpg".to_string(),
        ],
    };

    let images = load_images(&vec_imgs, config.vision_config.image_size)?.to_device(&device)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_filename.clone()], DType::F32, &device)?
    };

    let model = clip::ClipModel::new(vb, &config)?;
    let (input_ids, vec_seq) = tokenize_sequences(args.sequences, &tokenizer, &device)?;
    let (_logits_per_text, logits_per_image) = model.forward(&images, &input_ids)?;
    let softmax_image = softmax(&logits_per_image, 1)?;
    let softmax_image_vec = softmax_image.flatten_all()?.to_vec1::<f32>()?;
    println!("softmax_image_vec: {:?}", softmax_image_vec);
    let probability_vec = softmax_image_vec
        .iter()
        .map(|v| v * 100.0)
        .collect::<Vec<f32>>();
    let probability_per_image = probability_vec.len() / vec_imgs.len();
    for (i, img) in vec_imgs.iter().enumerate() {
        let start = i * probability_per_image;
        let end = start + probability_per_image;
        let prob = &probability_vec[start..end];
        println!("\n\nResults for image: {}\n", img);
        for (i, p) in prob.iter().enumerate() {
            println!("Probability: {:.4}% Text: {} ", p, vec_seq[i]);
        }
    }
    Ok(())
}

pub fn tokenize_sequences(
    sequences: Option<Vec<String>>,
    tokenizer: &Tokenizer,
    device: &Device,
) -> anyhow::Result<(Tensor, Vec<String>)> {
    let pad_id = *tokenizer
        .get_vocab(true)
        .get("<|endoftext|>")
        .ok_or(E::msg("No pad token"))?;
    let vec_seq = match sequences {
        Some(seq) => seq,
        None => vec![
            "a cycling race".to_string(),
            "a photo of two cats".to_string(),
            "a robot holding a candle".to_string(),
        ],
    };
    let mut tokens = vec![];
    for seq in vec_seq.clone() {
        let encoding = tokenizer.encode(seq, true).map_err(E::msg)?;
        tokens.push(encoding.get_ids().to_vec());
    }
    let max_len = tokens.iter().map(|v| v.len()).max().unwrap_or(0);
    // Pad the sequences to have the same length
    for token_vec in tokens.iter_mut() {
        let len_diff = max_len - token_vec.len();
        if len_diff > 0 {
            token_vec.extend(vec![pad_id; len_diff]);
        }
    }
    let input_ids = Tensor::new(tokens, device)?;
    Ok((input_ids, vec_seq))
}
