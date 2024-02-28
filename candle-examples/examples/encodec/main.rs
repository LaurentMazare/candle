#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::encodec::{Config, Model};
use clap::Parser;
use hf_hub::api::sync::Api;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// The model weight file, in safetensor format.
    #[arg(long)]
    model: Option<String>,

    /// Input file as a safetensors containing the encodec tokens.
    #[arg(long)]
    code_file: String,

    /// Output file that will be generated in wav format.
    #[arg(long)]
    out: String,

    /// Do another step of encoding the PCM data and and decoding the resulting codes.
    #[arg(long)]
    roundtrip: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let device = candle_examples::device(args.cpu)?;
    let model = match args.model {
        Some(model) => std::path::PathBuf::from(model),
        None => Api::new()?
            .model("facebook/encodec_24khz".to_string())
            .get("model.safetensors")?,
    };
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model], DType::F32, &device)? };
    let config = Config::default();
    let model = Model::new(&config, vb)?;

    let codes = candle::safetensors::load(args.code_file, &device)?;
    let codes = codes.get("codes").expect("no codes in input file").i(0)?;
    println!("codes shape: {:?}", codes.shape());
    let pcm = model.decode(&codes)?;
    println!("pcm shape: {:?}", pcm.shape());

    let pcm = if args.roundtrip {
        let codes = model.encode(&pcm)?;
        println!("second step codes shape: {:?}", pcm.shape());
        let pcm = model.decode(&codes)?;
        println!("second step pcm shape: {:?}", pcm.shape());
        pcm
    } else {
        pcm
    };

    let pcm = pcm.i(0)?.i(0)?.to_vec1::<f32>()?;
    let mut output = std::fs::File::create(&args.out)?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, 24_000)?;

    Ok(())
}
