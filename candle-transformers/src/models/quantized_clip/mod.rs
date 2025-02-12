use candle::{Result, Tensor, D};

use crate::models::clip;
use crate::models::clip::text_model::{ClipTextConfig, Activation};
use crate::models::clip::vision_model::ClipVisionConfig;
use crate::quantized_nn as quantized_nn;
use crate::quantized_var_builder::VarBuilder;

pub mod text_model;
pub mod vision_model;

#[derive(Clone, Debug)]
pub enum EncoderConfig {
    Text(ClipTextConfig),
    Vision(ClipVisionConfig),
}

impl EncoderConfig {
    pub fn embed_dim(&self) -> usize {
        match self {
            Self::Text(c) => c.embed_dim,
            Self::Vision(c) => c.embed_dim,
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
            Self::Vision(c) => c.activation,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClipModel {
    text_model: text_model::ClipTextTransformer,
    vision_model: vision_model::ClipVisionTransformer,
    visual_projection: quantized_nn::Linear,
    text_projection: quantized_nn::Linear,
    logit_scale: Tensor,
}

impl ClipModel {
    pub fn new(vs: VarBuilder, c: &clip::ClipConfig) -> Result<Self> {
        let text_model = text_model::ClipTextTransformer::new(vs.pp("text_model"), &c.text_config)?;

        let vision_model = vision_model::ClipVisionTransformer::new(vs.pp("vision_model"), &c.vision_config)?;

        let visual_projection = quantized_nn::linear_no_bias(
            c.vision_config.embed_dim,
            c.vision_config.projection_dim,
            vs.pp("visual_projection"),
        )?;

        let text_projection = quantized_nn::linear_no_bias(
            c.text_config.embed_dim,
            c.text_config.projection_dim,
            vs.pp("text_projection"),
        )?;

        // originally nn.Parameter
        let logit_scale = if vs.contains_key("logit_scale") {
            vs.get(&[], "logit_scale")?.dequantize(vs.device())?
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
