use std::collections::HashMap;

use crate::models::{
    clip::{text_model::Activation, vision_model::ClipVisionConfig},
    llama::{Config, LlamaEosToks},
};
use serde::{Deserialize, Serialize};

// original config from liuhaotian/llava
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LLaVAConfig {
    pub architectures: Vec<String>,
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub hidden_size: usize,
    #[serde(default = "default_image_aspect_ratio")]
    pub image_aspect_ratio: String,
    pub image_crop_resolution: usize,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub image_split_resolution: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub mm_hidden_size: usize,
    #[serde(default = "default_mm_patch_merge_type")]
    pub mm_patch_merge_type: String,
    pub mm_projector_type: String,
    pub mm_use_im_start_end: bool,
    pub mm_vision_select_feature: String,
    pub mm_vision_select_layer: isize,
    pub mm_vision_tower: Option<String>,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub pad_token_id: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub tokenizer_model_max_length: Option<usize>,
    pub torch_dtype: String,
    pub use_cache: bool,
    pub vocab_size: usize,
    #[serde(default = "default_image_token_index")]
    pub image_token_index: isize,
    #[serde(default = "default_hf")]
    pub hf: bool,
    pub tie_word_embeddings: Option<bool>,
}

fn default_hf() -> bool {
    false
}

fn default_image_token_index() -> isize {
    -200
}

fn default_mm_patch_merge_type() -> String {
    "flat".to_string()
}

fn default_image_aspect_ratio() -> String {
    "square".to_string()
}

impl LLaVAConfig {
    pub fn to_llama_config(&self) -> Config {
        Config {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            vocab_size: self.vocab_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps as f64,
            rope_theta: self.rope_theta,
            bos_token_id: Some(self.bos_token_id as u32),
            eos_token_id: Some(LlamaEosToks::Single(self.eos_token_id as u32)),
            use_flash_attn: false,
            rope_scaling: None, // Assume we don't have LLaVA for Llama 3.1
            max_position_embeddings: self.max_position_embeddings,
            tie_word_embeddings: self.tie_word_embeddings.unwrap_or(false),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFLLaVATextConfig {
    pub architectures: Vec<String>,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    pub max_position_embeddings: usize,
    pub model_type: String,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_key_value_heads")]
    pub num_key_value_heads: usize,
    pub pad_token_id: usize,
    pub rms_norm_eps: f32,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    pub torch_dtype: String,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub vocab_size: usize,
}

fn default_num_hidden_layers() -> usize {
    32
}

fn default_use_cache() -> bool {
    true
}

fn default_hidden_size() -> usize {
    4096
}

fn default_intermediate_size() -> usize {
    11008
}

fn default_max_length() -> usize {
    4096
}

fn default_num_attention_heads() -> usize {
    32
}

fn default_num_key_value_heads() -> usize {
    32
}

fn default_rope_theta() -> f32 {
    10000.0
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFLLaVAVisionConfig {
    pub hidden_size: usize,
    pub image_size: usize,
    pub intermediate_size: usize,
    pub model_type: String,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub patch_size: usize,
    pub projection_dim: usize,
    pub vocab_size: usize,
}

// config from llava-v1.6-vicuna-7b-hf
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFLLaVAConfig {
    pub architectures: Vec<String>,
    pub ignore_index: isize,
    pub image_grid_pinpoints: Vec<(u32, u32)>,
    pub image_token_index: isize,
    pub model_type: String,
    pub projector_hidden_act: String,
    pub text_config: HFLLaVATextConfig,
    pub torch_dtype: String,
    pub use_image_newline_parameter: bool,
    pub vision_config: HFLLaVAVisionConfig,
    pub vision_feature_layer: isize,
    pub vision_feature_select_strategy: String,
    pub vocab_size: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFGenerationConfig {
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    pub pad_token_id: usize,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct HFPreProcessorConfig {
    pub aspect_ratio_setting: String,
    pub crop_size: HashMap<String, usize>,
    pub do_center_crop: bool,
    pub do_convert_rgb: bool,
    pub do_normalize: bool,
    pub do_rescale: bool,
    pub do_resize: bool,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub resample: u32,
    pub rescale_factor: f32,
    pub size: HashMap<String, f32>,
}

impl HFLLaVAConfig {
    pub fn to_clip_vision_config(&self) -> ClipVisionConfig {
        ClipVisionConfig {
            hidden_size: self.vision_config.hidden_size,
            hidden_act: Activation::QuickGelu,
            intermediate_size: self.vision_config.intermediate_size,
            num_hidden_layers: self.vision_config.num_hidden_layers,
            num_attention_heads: self.vision_config.num_attention_heads,
            projection_dim: self.vision_config.projection_dim,
            num_channels: 3,
            image_size: self.vision_config.image_size,
            patch_size: self.vision_config.patch_size,
        }
    }
    fn map_projector_type(s: &str) -> String {
        if s == "gelu" {
            "mlp2x_gelu".to_string()
        } else {
            s.to_string()
        }
    }

    fn map_select_feature(s: &str) -> String {
        if s == "default" {
            "patch".to_string()
        } else {
            "cls_patch".to_string()
        }
    }

    pub fn to_llava_config(
        &self,
        generation_config: &HFGenerationConfig,
        preprocessor_config: &HFPreProcessorConfig,
    ) -> LLaVAConfig {
        LLaVAConfig {
            hf: true,
            architectures: self.architectures.clone(),
            bos_token_id: generation_config.bos_token_id,
            eos_token_id: generation_config.eos_token_id,
            hidden_size: self.text_config.hidden_size,
            image_aspect_ratio: preprocessor_config.aspect_ratio_setting.clone(),
            image_crop_resolution: 224,
            image_grid_pinpoints: self.image_grid_pinpoints.clone(),
            image_split_resolution: 224,
            intermediate_size: self.text_config.intermediate_size,
            max_position_embeddings: self.text_config.max_position_embeddings,
            mm_hidden_size: 1024,
            mm_patch_merge_type: "spatial_unpad".to_string(),
            mm_projector_type: Self::map_projector_type(&self.projector_hidden_act),
            mm_use_im_start_end: false,
            mm_vision_select_feature: Self::map_select_feature(
                &self.vision_feature_select_strategy,
            ),
            mm_vision_select_layer: self.vision_feature_layer,
            mm_vision_tower: None,
            model_type: self.model_type.clone(),
            num_attention_heads: self.text_config.num_attention_heads,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_key_value_heads: self.text_config.num_key_value_heads,
            pad_token_id: self.text_config.pad_token_id,
            rms_norm_eps: self.text_config.rms_norm_eps,
            rope_theta: self.text_config.rope_theta,
            tokenizer_model_max_length: Some(4096),
            torch_dtype: self.torch_dtype.clone(),
            use_cache: self.text_config.use_cache,
            vocab_size: self.vocab_size,
            image_token_index: self.image_token_index,
            tie_word_embeddings: None,
        }
    }
}
