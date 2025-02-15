use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
pub enum Llama3RopeType {
    #[serde(rename = "llama3")]
    Llama3,
    #[default]
    #[serde(rename = "default")]
    Default,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
pub struct Llama3RopeConfig {
    pub factor: f32,
    pub low_freq_factor: f32,
    pub high_freq_factor: f32,
    pub original_max_position_embeddings: usize,
    pub rope_type: Llama3RopeType,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MllamaTextConfig {
    #[serde(default = "MllamaTextConfig::vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "MllamaTextConfig::hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "MllamaTextConfig::hidden_act")]
    pub hidden_act: String,
    #[serde(default = "MllamaTextConfig::num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "MllamaTextConfig::num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "MllamaTextConfig::num_key_value_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "MllamaTextConfig::intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "MllamaTextConfig::rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "MllamaTextConfig::rope_scaling")]
    pub rope_scaling: Option<Llama3RopeConfig>,
    #[serde(default = "MllamaTextConfig::rms_norm_eps")]
    pub rms_norm_eps: f32,
    #[serde(default = "MllamaTextConfig::max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "MllamaTextConfig::initializer_range")]
    pub initializer_range: f32,
    #[serde(default = "MllamaTextConfig::use_cache")]
    pub use_cache: bool,
    #[serde(default = "MllamaTextConfig::tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "MllamaTextConfig::cross_attention_layers")]
    pub cross_attention_layers: Option<Vec<usize>>,
    #[serde(default = "MllamaTextConfig::dropout")]
    pub dropout: f32,
    #[serde(default = "MllamaTextConfig::bos_token_id")]
    pub bos_token_id: usize,
    #[serde(default = "MllamaTextConfig::eos_token_id")]
    pub eos_token_id: usize,
    #[serde(default = "MllamaTextConfig::pad_token_id")]
    pub pad_token_id: Option<usize>,
}
impl MllamaTextConfig {
    fn vocab_size() -> usize {
        128256
    }
    fn hidden_size() -> usize {
        4096
    }
    fn hidden_act() -> String {
        String::from("silu")
    }
    fn num_hidden_layers() -> usize {
        40
    }
    fn num_attention_heads() -> usize {
        32
    }
    fn num_key_value_heads() -> usize {
        8
    }
    fn intermediate_size() -> usize {
        14_336
    }
    fn rope_theta() -> f32 {
        500000.0
    }
    fn rope_scaling() -> Option<Llama3RopeConfig> {
        None
    }
    fn rms_norm_eps() -> f32 {
        1e-5
    }
    fn max_position_embeddings() -> usize {
        131_072
    }
    fn initializer_range() -> f32 {
        0.02
    }
    fn use_cache() -> bool {
        true
    }
    fn tie_word_embeddings() -> bool {
        false
    }
    fn cross_attention_layers() -> Option<Vec<usize>> {
        None
    }
    fn dropout() -> f32 {
        0.0
    }
    fn bos_token_id() -> usize {
        128000
    }
    fn eos_token_id() -> usize {
        128001
    }
    fn pad_token_id() -> Option<usize> {
        Some(128004)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MllamaVisionConfig {
    #[serde(default = "MllamaVisionConfig::hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "MllamaVisionConfig::hidden_act")]
    pub hidden_act: String,

    #[serde(default = "MllamaVisionConfig::num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "MllamaVisionConfig::num_global_layers")]
    pub num_global_layers: usize,

    #[serde(default = "MllamaVisionConfig::num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "MllamaVisionConfig::num_channels")]
    pub num_channels: usize,

    #[serde(default = "MllamaVisionConfig::intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "MllamaVisionConfig::vision_output_dim")]
    pub vision_output_dim: usize,

    #[serde(default = "MllamaVisionConfig::image_size")]
    pub image_size: usize,

    #[serde(default = "MllamaVisionConfig::patch_size")]
    pub patch_size: usize,

    #[serde(default = "MllamaVisionConfig::norm_eps")]
    pub norm_eps: f32,

    #[serde(default = "MllamaVisionConfig::max_num_tiles")]
    pub max_num_tiles: usize,

    #[serde(default = "MllamaVisionConfig::intermediate_layers_indices")]
    pub intermediate_layers_indices: Vec<i32>,

    #[serde(default = "MllamaVisionConfig::supported_aspect_ratios")]
    pub supported_aspect_ratios: Vec<Vec<i32>>,

    #[serde(default = "MllamaVisionConfig::initializer_range")]
    pub initializer_range: f32,
}
impl MllamaVisionConfig {
    fn hidden_size() -> usize {
        1280
    }

    fn hidden_act() -> String {
        String::from("gelu")
    }

    fn num_hidden_layers() -> usize {
        32
    }

    fn num_global_layers() -> usize {
        8
    }

    fn num_attention_heads() -> usize {
        16
    }

    fn num_channels() -> usize {
        3
    }

    fn intermediate_size() -> usize {
        5120
    }

    fn vision_output_dim() -> usize {
        7680
    }

    fn image_size() -> usize {
        448
    }

    fn patch_size() -> usize {
        14
    }

    fn norm_eps() -> f32 {
        1e-5
    }

    fn max_num_tiles() -> usize {
        4
    }

    fn intermediate_layers_indices() -> Vec<i32> {
        vec![3, 7, 15, 23, 30]
    }

    fn supported_aspect_ratios() -> Vec<Vec<i32>> {
        vec![
            vec![1, 1],
            vec![1, 2],
            vec![1, 3],
            vec![1, 4],
            vec![2, 1],
            vec![2, 2],
            vec![3, 1],
            vec![4, 1],
        ]
    }

    fn initializer_range() -> f32 {
        0.02
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MllamaConfig {
    pub vision_config: MllamaVisionConfig,
    pub text_config: MllamaTextConfig,
    #[serde(default = "MllamaConfig::image_token_index")]
    pub image_token_index: i32,
}
impl MllamaConfig {
    fn image_token_index() -> i32 {
        128256
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ImagePreProcessorConfig {
    #[serde(default = "ImagePreProcessorConfig::do_convert_rgb")]
    pub do_convert_rgb: bool,
    #[serde(default = "ImagePreProcessorConfig::do_normalize")]
    pub do_normalize: bool,
    #[serde(default = "ImagePreProcessorConfig::do_pad")]
    pub do_pad: bool,
    #[serde(default = "ImagePreProcessorConfig::do_rescale")]
    pub do_rescale: bool,
    #[serde(default = "ImagePreProcessorConfig::do_resize")]
    pub do_resize: bool,
    #[serde(default = "ImagePreProcessorConfig::image_mean")]
    pub image_mean: Vec<f32>,
    #[serde(default = "ImagePreProcessorConfig::image_std")]
    pub image_std: Vec<f32>,
    #[serde(default = "ImagePreProcessorConfig::max_image_tiles")]
    pub max_image_tiles: usize,
    #[serde(default = "ImagePreProcessorConfig::resample")]
    pub resample: usize,
    #[serde(default = "ImagePreProcessorConfig::rescale_factor")]
    pub rescale_factor: f32,
    #[serde(default = "ImagePreProcessorConfig::size")]
    pub size: HashMap<String, usize>,
}
impl ImagePreProcessorConfig {
    fn do_convert_rgb() -> bool {
        true
    }
    fn do_normalize() -> bool {
        true
    }
    fn do_pad() -> bool {
        true
    }
    fn do_rescale() -> bool {
        true
    }
    fn do_resize() -> bool {
        true
    }
    fn image_mean() -> Vec<f32> {
        vec![0.48145466, 0.4578275, 0.40821073]
    }
    fn image_std() -> Vec<f32> {
        vec![0.26862954, 0.26130258, 0.27577711]
    }
    fn max_image_tiles() -> usize {
        4
    }
    fn resample() -> usize {
        2
    }
    fn rescale_factor() -> f32 {
        0.00392156862745098
    }
    fn size() -> HashMap<String, usize> {
        let mut size = HashMap::new();
        size.insert(String::from("hight"), 448);
        size.insert(String::from("width"), 448);
        size
    }
}
