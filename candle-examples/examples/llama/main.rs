// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use clap::{Parser, ValueEnum};

use candle::{DType, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::llama as model;
use hf_hub::{api::sync::Api, Repo, RepoType};
use model::{Llama, LlamaConfig};
use std::io::Write;
use std::path::Path;

const EOS_TOKEN: &str = "</s>";
const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    V1,
    V2,
    V3,
    V31,
    V3Instruct,
    V31Instruct,
    V32_1b,
    V32_1bInstruct,
    V32_3b,
    V32_3bInstruct,
    #[value(name = "solar-10.7b")]
    Solar10_7B,
    #[value(name = "tiny-llama-1.1b-chat")]
    TinyLlama1_1BChat,
    #[value(name = "SmoLM2-1.7B")]
    SmolLM2_1B,
    #[value(name = "SmoLM2-1.7B-Instruct")]
    SmolLM2_1BInstruct,
    #[value(name = "SmoLM2-360M")]
    SmolLM2_360M,
    #[value(name = "SmoLM2-360M-Instruct")]
    SmolLM2_360MInstruct,
    #[value(name = "SmoLM2-135M")]
    SmolLM2_135M,
    #[value(name = "SmoLM2-135M-Instruct")]
    SmolLM2_135MInstruct,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// Only sample among the top K samples.
    #[arg(long)]
    top_k: Option<usize>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(short = 'n', long, default_value_t = 10000)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    /// Use different dtype than f16
    #[arg(long)]
    dtype: Option<String>,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    /// The model size to use.
    #[arg(long, default_value = "v3")]
    which: Which,

    #[arg(long)]
    use_flash_attn: bool,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f32,

    /// The context size to consider for the repeat penalty.
    #[arg(long, default_value_t = 128)]
    repeat_last_n: usize,

    #[arg(long)]
    weight_path: Option<String>,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();
    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };

    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::F16,
    };
    let (llama, tokenizer_filename, mut cache, config) = {
        let api = Api::new()?;
        let model_id = args.model_id.unwrap_or_else(|| {
            let str = match args.which {
                Which::V1 => "Narsil/amall-7b",
                Which::V2 => "meta-llama/Llama-2-7b-hf",
                Which::V3 => "meta-llama/Meta-Llama-3-8B",
                Which::V3Instruct => "meta-llama/Meta-Llama-3-8B-Instruct",
                Which::V31 => "meta-llama/Llama-3.1-8B",
                Which::V31Instruct => "meta-llama/Llama-3.1-8B-Instruct",
                Which::V32_1b => "meta-llama/Llama-3.2-1B",
                Which::V32_1bInstruct => "meta-llama/Llama-3.2-1B-Instruct",
                Which::V32_3b => "meta-llama/Llama-3.2-3B",
                Which::V32_3bInstruct => "meta-llama/Llama-3.2-3B-Instruct",
                Which::Solar10_7B => "upstage/SOLAR-10.7B-v1.0",
                Which::TinyLlama1_1BChat => "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                Which::SmolLM2_135M => "HuggingFaceTB/SmolLM2-135M",
                Which::SmolLM2_135MInstruct => "HuggingFaceTB/SmolLM2-135M-Instruct",
                Which::SmolLM2_360M => "HuggingFaceTB/SmolLM2-360M",
                Which::SmolLM2_360MInstruct => "HuggingFaceTB/SmolLM2-360M-Instruct",
                Which::SmolLM2_1B => "HuggingFaceTB/SmolLM2-1.7B",
                Which::SmolLM2_1BInstruct => "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            };
            str.to_string()
        });
        println!("loading the model weights from {model_id}");
        let revision = args.revision.unwrap_or("main".to_string());
        let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let tokenizer_filename = match &args.weight_path {
            Some(path) => Path::new(path).join("tokenizer.json"),
            _ => api.get("tokenizer.json")?,
        };
        let config_filename = match &args.weight_path {
            Some(path) => Path::new(path).join("config.json"),
            _ => api.get("config.json")?,
        };

        let config: LlamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
        let config = config.into_config(args.use_flash_attn);

        let filenames = match args.which {
            Which::V1
            | Which::V2
            | Which::V3
            | Which::V3Instruct
            | Which::V31
            | Which::V31Instruct
            | Which::V32_3b
            | Which::V32_3bInstruct
            | Which::Solar10_7B => match &args.weight_path {
                Some(path) => candle_examples::hub_load_local_safetensors(
                    path,
                    "model.safetensors.index.json",
                )?,
                _ => candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?,
            },
            Which::SmolLM2_360M
            | Which::SmolLM2_360MInstruct
            | Which::SmolLM2_135M
            | Which::SmolLM2_135MInstruct
            | Which::SmolLM2_1B
            | Which::SmolLM2_1BInstruct
            | Which::V32_1b
            | Which::V32_1bInstruct
            | Which::TinyLlama1_1BChat => match &args.weight_path {
                Some(path) => vec![Path::new(path).join("model.safetensors")],
                _ => vec![api.get("model.safetensors")?],
            },
        };
        let cache = model::Cache::new(!args.no_kv_cache, dtype, &config, &device)?;

        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };
        (Llama::load(vb, &config)?, tokenizer_filename, cache, config)
    };
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eos_token_id = config.eos_token_id.or_else(|| {
        tokenizer
            .token_to_id(EOS_TOKEN)
            .map(model::LlamaEosToks::Single)
    });
    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    print!("{prompt}");
    let mut logits_processor = {
        let temperature = args.temperature;
        let sampling = if temperature <= 0. {
            Sampling::ArgMax
        } else {
            match (args.top_k, args.top_p) {
                (None, None) => Sampling::All { temperature },
                (Some(k), None) => Sampling::TopK { k, temperature },
                (None, Some(p)) => Sampling::TopP { p, temperature },
                (Some(k), Some(p)) => Sampling::TopKThenTopP { k, p, temperature },
            }
        };
        LogitsProcessor::from_sampling(args.seed, sampling)
    };

    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..args.sample_len {
        let (context_size, context_index) = if cache.use_kv_cache && index > 0 {
            (1, index_pos)
        } else {
            (tokens.len(), 0)
        };
        if index == 1 {
            start_gen = std::time::Instant::now()
        }
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = llama.forward(&input, context_index, &mut cache)?;
        let logits = logits.squeeze(0)?;
        let logits = if args.repeat_penalty == 1. {
            logits
        } else {
            let start_at = tokens.len().saturating_sub(args.repeat_last_n);
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                args.repeat_penalty,
                &tokens[start_at..],
            )?
        };
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        match eos_token_id {
            Some(model::LlamaEosToks::Single(eos_tok_id)) if next_token == eos_tok_id => {
                break;
            }
            Some(model::LlamaEosToks::Multiple(ref eos_ids)) if eos_ids.contains(&next_token) => {
                break;
            }
            _ => (),
        }
        if let Some(t) = tokenizer.next_token(next_token)? {
            print!("{t}");
            std::io::stdout().flush()?;
        }
    }
    if let Some(rest) = tokenizer.decode_rest().map_err(E::msg)? {
        print!("{rest}");
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        (token_generated - 1) as f64 / dt.as_secs_f64(),
    );
    Ok(())
}
