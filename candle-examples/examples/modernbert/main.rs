use std::path::PathBuf;

use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert;
use clap::{Parser, ValueEnum};
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};
use candle_nn::ops::softmax;
use candle_transformers::models::modernbert::NERItem;

#[derive(Debug, Clone, ValueEnum)]
enum Model {
    ModernBertBase,
    ModernBertLarge,
}


#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    #[arg(long, default_value = "modern-bert-base")]
    model: Model,

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
        None => match args.model {
            Model::ModernBertBase => "answerdotai/ModernBERT-base".to_string(),
            Model::ModernBertLarge => "answerdotai/ModernBERT-large".to_string(),
        },
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
                    anyhow::bail!("Model weights not found. The weights should either be a `model.safetensors` or `pytorch_model.bin` file.  Error: {e}")
                }
            },
        },
    };

    let config = std::fs::read_to_string(config_filename)?;
    let config: modernbert::Config = serde_json::from_str(&config)?;
    let mut tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let device = candle_examples::device(args.cpu)?;

    let vb = if weights_filename.ends_with("model.safetensors") {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], candle::DType::F32, &device)
                .unwrap()
        }
    } else {
        println!("Loading weights from pytorch_model.bin");
        VarBuilder::from_pth(&weights_filename, candle::DType::F32, &device).unwrap()
    };
    tokenizer
        .with_padding(Some(PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: config.pad_token_id,
            ..Default::default()
        }))
        .with_truncation(None)
        .map_err(E::msg)?;

    let prompt = match &args.prompt {
        Some(p) => vec![p.as_str()],
        None => vec![
            "Hello I'm a [MASK] model.",
            "I'm a [MASK] boy.",
            "I'm [MASK] in berlin.",
            "The capital of France is [MASK].",
        ],
    };
    let model = modernbert::ModernBertForMaskedLM::load(vb.clone(), &config)?;

    let input_ids = tokenize_batch(&tokenizer, prompt.clone(), &device)?;
    let attention_mask = get_attention_mask(&tokenizer, prompt.clone(), &device)?;

    let output = model
        .forward(&input_ids, &attention_mask)?
        .to_dtype(candle::DType::F32)?;

    let max_outs = output.argmax(2)?;

    let max_out = max_outs.to_vec2::<u32>()?;
    let max_out_refs: Vec<&[u32]> = max_out.iter().map(|v| v.as_slice()).collect();
    let decoded = tokenizer.decode_batch(&max_out_refs, true).unwrap();
    for (i, sentence) in decoded.iter().enumerate() {
        println!("Sentence: {} : {}", i + 1, sentence);
    }

    // Token Classification Example
    println!("\nToken Classification Example");

    let weights_filename = "/Users/scampion/src/HF/piiranha/model.safetensors";
    let config_filename = "/Users/scampion/src/HF/piiranha/config.json";
    let text = "My email is john.doe@example.com and my phone number is 555-123-4567.";
    let text2 = "Salut, vous venez souvent ici ? Je m'appelle Jean-Claude et je suis un grand fan de ski.";
    let texts = vec![text, text2];
    let config = std::fs::read_to_string(config_filename)?;
    let config: modernbert::Config = serde_json::from_str(&config)?;
    let vb = if weights_filename.ends_with("model.safetensors") {
        unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], candle::DType::F32, &device)
                .unwrap()
        }
    } else {
        println!("Loading weights from pytorch_model.bin");
        VarBuilder::from_pth(&weights_filename, candle::DType::F32, &device).unwrap()
    };
    let input_ids = tokenize_batch(&tokenizer, texts.clone(), &device)?;
    let tokenizer_encodings = tokenizer.encode_batch(texts.clone(), true).unwrap();
    let attention_mask = get_attention_mask(&tokenizer, texts.clone(), &device)?;
    let model = modernbert::ModernBertForTokenClassification::load(vb, &config)?;
    let logits = model.forward(&input_ids, &attention_mask)?;

    // let num_matrices = logits.dim(0)?;
    // for i in 0..num_matrices {
    //     let logits_2d: Vec<Vec<f32>> = logits.get(i)?.to_vec2::<f32>()?;
    //     for row in logits_2d {
    //         for val in row {
    //             print!("{:.2} ", val);
    //         }
    //         println!();
    //     }
    //     println!("{}", "%".repeat(80));
    // }
    let max_scores_vec = softmax(&logits, 2)?.max(2)?.to_vec2::<f32>()?;
    let max_indices_vec: Vec<Vec<u32>> = logits.argmax(2)?.to_vec2()?;
    let input_ids = input_ids.to_vec2::<u32>()?;

    let id2label = config.id2label;

    for (input_row_idx, input_id_row) in input_ids.iter().enumerate() {
        println!("Text: {:?}", texts[input_row_idx]);
        let mut current_row_result: Vec<NERItem> = Default::default();
        let current_row_encoding = tokenizer_encodings.get(input_row_idx).unwrap();
        let current_row_tokens = current_row_encoding.get_tokens();
        let current_row_max_scores = max_scores_vec.get(input_row_idx).unwrap();
        for (input_id_idx, _input_id) in input_id_row.iter().enumerate() {
            // Do not include special characters in output
            if current_row_encoding.get_special_tokens_mask()[input_id_idx] == 1 {
                continue;
            }
            let max_label_idx = max_indices_vec
                .get(input_row_idx)
                .unwrap()
                .get(input_id_idx)
                .unwrap();

            let label = id2label.clone().unwrap().get(max_label_idx).unwrap().clone();

            // Do not include those labeled as "O" ("Other")
            if label == "O" {
                continue;
            }

            current_row_result.push(NERItem {
                entity: label,
                word: current_row_tokens[input_id_idx].clone(),
                score: current_row_max_scores[input_id_idx],
                start: current_row_encoding.get_offsets()[input_id_idx].0,
                end: current_row_encoding.get_offsets()[input_id_idx].1,
                index: input_id_idx,
            });
        }
        println!("{:?}\n\n", current_row_result);
    }

    Ok(())
}

pub fn tokenize_batch(
    tokenizer: &Tokenizer,
    input: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(input, true).map_err(E::msg)?;

    let token_ids = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_ids().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle::Result<Vec<_>>>()?;

    Ok(Tensor::stack(&token_ids, 0)?)
}

pub fn get_attention_mask(
    tokenizer: &Tokenizer,
    input: Vec<&str>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let tokens = tokenizer.encode_batch(input, true).map_err(E::msg)?;

    let attention_mask = tokens
        .iter()
        .map(|tokens| {
            let tokens = tokens.get_attention_mask().to_vec();
            Tensor::new(tokens.as_slice(), device)
        })
        .collect::<candle::Result<Vec<_>>>()?;
    Ok(Tensor::stack(&attention_mask, 0)?)
}
