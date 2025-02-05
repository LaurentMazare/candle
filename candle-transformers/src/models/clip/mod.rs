//! Contrastive Language-Image Pre-Training
//!
//! Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! - 💻 [GH Link](https://github.com/openai/CLIP)
//! - 💻 Transformers Python [reference implementation](https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip)
//! - 🤗 [HF Model](https://huggingface.co/openai/clip-vit-large-patch14-336)
//!

use self::{
    text_model::{Activation, ClipTextTransformer},
    vision_model::ClipVisionTransformer,
};
use candle::{Result, Tensor, D};
use serde::Deserialize;

pub mod text_model;
pub mod vision_model;

#[derive(Clone, Debug)]
pub struct ClipModel {
    text_model: ClipTextTransformer,
    vision_model: ClipVisionTransformer,
    visual_projection: candle_nn::Linear,
    text_projection: candle_nn::Linear,
    logit_scale: Tensor,
}

#[derive(Clone, Debug)]
pub enum EncoderConfig {
    Text(text_model::ClipTextConfig),
    Vision(vision_model::ClipVisionConfig),
}

impl EncoderConfig {
    pub fn embed_dim(&self) -> usize {
        match self {
            Self::Text(c) => c.hidden_size,
            Self::Vision(c) => c.hidden_size,
        }
    }

    pub fn num_attention_heads(&self) -> usize {
        match self {
            Self::Text(c) => c.num_attention_heads,
            Self::Vision(c) => c.num_attention_heads,
        }
    }

    pub fn intermediate_size(&self) -> usize {
        match self {
            Self::Text(c) => c.intermediate_size,
            Self::Vision(c) => c.intermediate_size,
        }
    }

    pub fn num_hidden_layers(&self) -> usize {
        match self {
            Self::Text(c) => c.num_hidden_layers,
            Self::Vision(c) => c.num_hidden_layers,
        }
    }

    pub fn activation(&self) -> Activation {
        match self {
            Self::Text(_c) => Activation::QuickGelu,
            Self::Vision(c) => c.hidden_act,
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ClipConfig {
    pub text_config: text_model::ClipTextConfig,
    pub vision_config: vision_model::ClipVisionConfig,
    pub logit_scale_init_value: f32,
}

impl ClipConfig {
    // base image size is 224, model size is 600Mb
    pub fn vit_base_patch32() -> Self {
        let text_config = text_model::ClipTextConfig::vit_base_patch32();
        let vision_config = vision_model::ClipVisionConfig::vit_base_patch32();

        Self {
            text_config,
            vision_config,
            logit_scale_init_value: 2.6592,
        }
    }
}

impl ClipModel {
    pub fn new(vs: candle_nn::VarBuilder, c: &ClipConfig) -> Result<Self> {
        let text_model = ClipTextTransformer::new(vs.pp("text_model"), &c.text_config)?;
        let vision_model = ClipVisionTransformer::new(vs.pp("vision_model"), &c.vision_config)?;
        let visual_projection = candle_nn::linear_no_bias(
            c.vision_config.hidden_size,
            c.vision_config.projection_dim,
            vs.pp("visual_projection"),
        )?;
        let text_projection = candle_nn::linear_no_bias(
            c.text_config.hidden_size,
            c.text_config.projection_dim,
            vs.pp("text_projection"),
        )?;
        // originally nn.Parameter
        let logit_scale = if vs.contains_tensor("logit_scale") {
            vs.get(&[], "logit_scale")?
        } else {
            Tensor::new(&[c.logit_scale_init_value], vs.device())?
        };
        Ok(Self {
            text_model,
            vision_model,
            visual_projection,
            text_projection,
            logit_scale,
        })
    }

    pub fn get_text_features(&self, input_ids: &Tensor) -> Result<Tensor> {
        input_ids
            .apply(&self.text_model)?
            .apply(&self.text_projection)
    }

    pub fn get_image_features(&self, pixel_values: &Tensor) -> Result<Tensor> {
        pixel_values
            .apply(&self.vision_model)?
            .apply(&self.visual_projection)
    }

    pub fn forward(&self, pixel_values: &Tensor, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        let image_features = self.get_image_features(pixel_values)?;
        let text_features = self.get_text_features(input_ids)?;
        let image_features_normalized = div_l2_norm(&image_features)?;
        let text_features_normalized = div_l2_norm(&text_features)?;
        let logits_per_text = text_features_normalized.matmul(&image_features_normalized.t()?)?;
        let logit_scale = self.logit_scale.exp()?;
        let logits_per_text = logits_per_text.broadcast_mul(&logit_scale)?;
        let logits_per_image = logits_per_text.t()?;
        Ok((logits_per_text, logits_per_image))
    }
}

pub fn div_l2_norm(v: &Tensor) -> Result<Tensor> {
    let l2_norm = v.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    v.broadcast_div(&l2_norm)
}
