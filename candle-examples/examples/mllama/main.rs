// TODO: we need to impelement these layers
// 1. MllamaPrecomputedPositionEmbedding
// 2. MllamaPrecomputedAspectRatioEmbedding
// 3. MllamaVisionSdpaAttention -> we have a struct named SdpaAttention, could we use it here too?
// 4. MllamaTextCrossSdpaAttention
// 5. MllamaTextSelfSdpaAttention

mod config;
mod image_processor;
mod language_model;
mod mllama_conditional_generation;
mod processing_mllama;
mod vision_model;

use crate::config::ImagePreProcessorConfig;
use crate::image_processor::ImageProcessor;
use crate::mllama_conditional_generation::MllamaForConditionalGeneration;
use crate::processing_mllama::MllamaProcessor;
use anyhow::Ok;
use anyhow::{Error as E, Result};
use candle::DType;
use candle_nn::VarBuilder;
use config::MllamaConfig;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

fn main() -> Result<()> {
    let api = Api::new()?;
    let revision = String::from("main");
    let api = api.repo(Repo::with_revision(
        String::from("meta-llama/Llama-3.2-11B-Vision"),
        RepoType::Model,
        revision,
    ));
    // only for test
    let prompt = String::from("<|image|><|begin_of_text|>If I had to write a haiku for this one");

    // Load config
    let config_filename = api.get("config.json")?;
    let config: MllamaConfig = serde_json::from_slice(&std::fs::read(config_filename)?)?;
    println!("config: {:?}", config);

    // Load Image Processor
    let preprocessor_config_filename = api.get("preprocessor_config.json")?;
    let preprocessor_config: ImagePreProcessorConfig =
        serde_json::from_slice(&std::fs::read(preprocessor_config_filename)?)?;
    let image_processor = ImageProcessor::new(&preprocessor_config);

    let img_path = String::from("/candle/candle-examples/examples/mllama/rabbit.jpg");
    let img = image::ImageReader::open(img_path)?.decode()?;

    // image_processor.preprocess(&img)?;

    // Load tokenizer
    let tokenizer_filename = api.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let mllama_processor = MllamaProcessor::new(image_processor, tokenizer);
    let inputs = mllama_processor.process(img, prompt)?;

    // Load weights
    let dtype = DType::F16;
    let device = candle_examples::device(true)?;
    let filenames = candle_examples::hub_load_safetensors(&api, "model.safetensors.index.json")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&filenames, dtype, &device)? };

    let mllama_model = MllamaForConditionalGeneration::new(vb, &config)?;

    println!("here we are");
    Ok(())
}
