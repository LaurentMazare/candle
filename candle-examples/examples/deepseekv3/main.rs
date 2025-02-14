// An implementation of LLaMA https://github.com/facebookresearch/llama
//
// This is based on nanoGPT in a similar way to:
// https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
//
// The tokenizer config can be retrieved from:
// https://huggingface.co/hf-internal-testing/llama-tokenizer/raw/main/tokenizer.json

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

use anyhow::{bail, Error as E, Result};
use candle::quantized::GgmlDType;
use clap::{Parser, ValueEnum};

use candle::{DType, Device, Tensor};
use candle_transformers::generation::LogitsProcessor;
use cudarc::driver::safe::CudaDevice;
use cudarc::nccl::safe::{Comm, Id};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::io::Write;
use std::rc::Rc;

mod model;
mod ops;
mod quant;
use model::{DeepSeekV3, DeepSeekV3Config};

const DEFAULT_PROMPT: &str = "My favorite theorem is ";

#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum)]
enum Which {
    R1,
    V3,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    num_shards: usize,

    #[arg(long)]
    rank: Option<usize>,

    /// The temperature used to generate samples.
    #[arg(long, default_value_t = 0.8)]
    temperature: f64,

    /// Nucleus sampling probability cutoff.
    #[arg(long)]
    top_p: Option<f64>,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The length of the sample to generate (in tokens).
    #[arg(long, default_value_t = 100)]
    sample_len: usize,

    /// Disable the key-value cache.
    #[arg(long)]
    no_kv_cache: bool,

    /// The initial prompt.
    #[arg(long)]
    prompt: Option<String>,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long)]
    revision: Option<String>,

    #[arg(long)]
    dtype: Option<String>,

    /// Quantization to apply to the model for faster inference. Defaults to q4k. One of: q2k,q3k,q4k,q5k,q8_0
    #[arg(long)]
    quant: Option<String>,

    #[arg(long, default_value = "r1")]
    which: Which,

    #[arg(long, default_value = "nccl_id.txt")]
    comm_file: String,
}

fn main() -> Result<()> {
    use tokenizers::Tokenizer;

    let args = Args::parse();

    let dtype = match args.dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(dtype) => bail!("Unsupported dtype {dtype}"),
        None => DType::BF16,
    };

    let comm_file = std::path::PathBuf::from(&args.comm_file);
    if comm_file.exists() {
        bail!("comm file {comm_file:?} already exists, please remove it first")
    }

    let api = Api::new()?;
    let model_id = match args.model_id {
        Some(model) => model,
        None => match args.which {
            Which::V3 => "deepseek-ai/DeepSeek-V3".to_string(),
            Which::R1 => "deepseek-ai/DeepSeek-R1".to_string(),
        },
    };
    println!("loading the model weights from {model_id}");
    let revision = args.revision.unwrap_or("main".to_string());
    let api = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));
    let config_filename = api.get("config.json")?;
    let config: DeepSeekV3Config = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    let tokenizer_filename = api.get("tokenizer.json")?;
    let filenames = candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;

    let rank = match args.rank {
        None => {
            println!("creating {} child processes", args.num_shards);
            let children: Vec<_> = (0..args.num_shards)
                .map(|rank| {
                    let mut args: std::collections::VecDeque<_> = std::env::args().collect();
                    args.push_back("--rank".to_string());
                    args.push_back(format!("{rank}"));
                    let name = args.pop_front().unwrap();
                    std::process::Command::new(name).args(args).spawn().unwrap()
                })
                .collect();
            for mut child in children {
                child.wait()?;
            }
            return Ok(());
        }
        Some(rank) => rank,
    };

    let num_shards = args.num_shards;
    // Primitive IPC
    let id = if rank == 0 {
        let id = Id::new().unwrap();
        let tmp_file = comm_file.with_extension(".comm.tgz");
        std::fs::File::create(&tmp_file)?
            .write_all(&id.internal().iter().map(|&i| i as u8).collect::<Vec<_>>())?;
        std::fs::rename(&tmp_file, &comm_file)?;
        id
    } else {
        while !comm_file.exists() {
            std::thread::sleep(std::time::Duration::from_secs(1));
        }
        let data = std::fs::read(&comm_file)?;
        let internal: [i8; 128] = data
            .into_iter()
            .map(|i| i as i8)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let id: Id = Id::uninit(internal);
        id
    };
    let device = CudaDevice::new(rank)?;
    let comm = match Comm::from_rank(device, rank, num_shards, id) {
        Ok(comm) => Rc::new(comm),
        Err(err) => anyhow::bail!("nccl error {:?}", err.0),
    };
    if rank == 0 {
        std::fs::remove_file(comm_file)?;
    }
    println!("Rank {rank:?} spawned");

    let device = Device::new_cuda(rank)?;

    let quant = match args.quant {
        Some(x) => match x.to_lowercase().as_str() {
            "q2k" => GgmlDType::Q2K,
            "q3k" => GgmlDType::Q3K,
            "q4k" => GgmlDType::Q4K,
            "q5k" => GgmlDType::Q5K,
            "q8_0" => GgmlDType::Q8_0,
            other => {
                anyhow::bail!("Quantization {other} is not supported, try q2k,q3k,q4k,q5k,q8_0")
            }
        },
        None => GgmlDType::Q4K,
    };

    println!("building the model");
    let vb = unsafe {
        candle_nn::var_builder::ShardedSafeTensors::var_builder(&filenames, dtype, &device)?
    };
    let mut model = DeepSeekV3::new(&config, vb, Some(quant), comm)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let prompt = args.prompt.as_ref().map_or(DEFAULT_PROMPT, |p| p.as_str());
    let mut tokens = tokenizer
        .encode(prompt, true)
        .map_err(E::msg)?
        .get_ids()
        .to_vec();
    let mut tokenizer = candle_examples::token_output_stream::TokenOutputStream::new(tokenizer);

    println!("starting the inference loop");
    let temperature = if args.temperature <= 0. {
        None
    } else {
        Some(args.temperature)
    };
    let mut logits_processor = LogitsProcessor::new(args.seed, temperature, args.top_p);
    let mut new_tokens = vec![];
    let mut start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    for index in 0..args.sample_len {
        // Only start timing at the second token as processing the first token waits for all the
        // weights to be loaded in an async way.
        if index == 1 {
            start_gen = std::time::Instant::now()
        };
        let context_size = if index > 0 { 1 } else { tokens.len() };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = model.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        tokens.push(next_token);
        new_tokens.push(next_token);
        if next_token == config.eos_token_id {
            break;
        }

        if rank == 0 {
            if let Some(t) = tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
    }
    println!();
    if rank == 0 {
        let dt = start_gen.elapsed();
        println!(
            "\n\n{} tokens generated ({} token/s)\n",
            args.sample_len,
            (args.sample_len - 1) as f64 / dt.as_secs_f64(),
        );
    }
    Ok(())
}
