use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{self, BertForMaskedLM, Config};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model to use, check out available models: https://huggingface.co/models?library=sentence-transformers&sort=trending
    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    // Path to the tokenizer file.
    #[arg(long)]
    tokenizer_file: Option<String>,

    // Path to the weight files.
    #[arg(long)]
    weight_files: Option<String>,

    // Path to the config file.
    #[arg(long)]
    config_file: Option<String>,

    /// When set, compute embeddings for this prompt.
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let api = Api::new()?;
    let model_id = match &args.model_id {
        Some(model_id) => model_id.to_string(),
        None => "prithivida/Splade_PP_en_v1".to_string(),
    };
    let repo = api.repo(Repo::with_revision(
        model_id,
        RepoType::Model,
        args.revision,
    ));

    let tokenizer_filename = match args.tokenizer_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("tokenizer.json")?,
    };

    let config_filename = match args.config_file {
        Some(file) => std::path::PathBuf::from(file),
        None => repo.get("config.json")?,
    };

    let weights_filename = match args.weight_files {
        Some(files) => PathBuf::from(files),
        None => match repo.get("model.safetensors") {
            Ok(safetensors) => safetensors,
            Err(_) => match repo.get("pytorch_model.bin") {
                Ok(pytorch_model) => pytorch_model,
                Err(e) => {
                    return Err(anyhow::Error::msg(format!("Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {}", e)));
                }
            },
        },
    };

    let config = std::fs::read_to_string(config_filename)?;
    let config: Config = serde_json::from_str(&config)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let device = candle_examples::device(args.cpu)?;
    let dtype = bert::DTYPE;

    let vb = if weights_filename.ends_with("model.safetensors") {
        unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device).unwrap() }
    } else {
        println!("Loading weights from pytorch_model.bin");
        VarBuilder::from_pth(&weights_filename, dtype, &device).unwrap()
    };
    let model = BertForMaskedLM::load(vb, &config, false)?;

    if let Some(prompt) = args.prompt {
        let tokenizer = tokenizer
            .with_padding(None)
            .with_truncation(None)
            .map_err(E::msg)?;
        let tokens = tokenizer
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let ys = model.forward(&token_ids, &token_type_ids, None)?;
        let vec = Tensor::log(
            &Tensor::try_from(1.0)?
                .to_dtype(dtype)?
                .to_device(&device)?
                .broadcast_add(&ys.relu()?)?,
        )?
        .max(1)?;
        let vec = normalize_l2(&vec)?;

        let vec = vec.squeeze(0)?.to_vec1::<f32>()?;

        let indices = (0..vec.len())
            .filter(|&i| vec[i] != 0.0)
            .map(|x| x as u32)
            .collect::<Vec<_>>();

        let tokens = tokenizer.decode(&indices, true).unwrap();
        println!("{tokens:?}");
        let values = indices.iter().map(|&i| vec[i as usize]).collect::<Vec<_>>();
        println!("{values:?}");
    } else {
        let sentences = [
            "The cat sits outside",
            "A man is playing guitar",
            "I love pasta",
            "The new movie is awesome",
            "The cat plays in the garden",
            "A woman watches TV",
            "The new movie is so great",
            "Do you like pizza?",
        ];

        let n_sentences = sentences.len();
        if let Some(pp) = tokenizer.get_padding_mut() {
            pp.strategy = tokenizers::PaddingStrategy::BatchLongest
        } else {
            let pp = PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            };
            tokenizer.with_padding(Some(pp));
        }
        let tokens = tokenizer
            .encode_batch(sentences.to_vec(), true)
            .map_err(E::msg)?;
        let token_ids = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_ids().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &device)?)
            })
            .collect::<Result<Vec<_>>>()?;
        let attention_mask = tokens
            .iter()
            .map(|tokens| {
                let tokens = tokens.get_attention_mask().to_vec();
                Ok(Tensor::new(tokens.as_slice(), &device)?)
            })
            .collect::<Result<Vec<_>>>()?;

        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let ys = model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;
        let vector = Tensor::log(
            &Tensor::try_from(1.0)?
                .to_dtype(dtype)?
                .to_device(&device)?
                .broadcast_add(&ys.relu()?)?,
        )?;
        let vector = vector
            .broadcast_mul(&attention_mask.unsqueeze(2)?.to_dtype(dtype)?)?
            .max(1)?;
        let vec = normalize_l2(&vector)?;
        let mut similarities = vec![];
        for i in 0..n_sentences {
            let e_i = vec.get(i)?;
            for j in (i + 1)..n_sentences {
                let e_j = vec.get(j)?;
                let sum_ij = (&e_i * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let sum_i2 = (&e_i * &e_i)?.sum_all()?.to_scalar::<f32>()?;
                let sum_j2 = (&e_j * &e_j)?.sum_all()?.to_scalar::<f32>()?;
                let cosine_similarity = sum_ij / (sum_i2 * sum_j2).sqrt();
                similarities.push((cosine_similarity, i, j))
            }
        }
        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));
        for &(score, i, j) in similarities[..5].iter() {
            println!("score: {score:.2} '{}' '{}'", sentences[i], sentences[j])
        }
    }

    Ok(())
}

pub fn normalize_l2(v: &Tensor) -> Result<Tensor> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}
