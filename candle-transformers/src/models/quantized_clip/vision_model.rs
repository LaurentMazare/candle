//! Contrastive Language-Image Pre-Training
//!
//! Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
//! pairs of images with related texts.
//!
//! https://github.com/openai/CLIP
//! https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip

use candle::{IndexOp, Result, Shape, Tensor, D};
use candle_nn as nn;
use candle_nn::Module;
use crate::quantized_nn;

use crate::models::clip;
use crate::models::clip::text_model::Activation;
use crate::quantized_var_builder::VarBuilder;
use nn::{Conv2d, Conv2dConfig};

use super::text_model::ClipEncoder;

#[derive(Debug, Clone)]
pub struct ClipVisionConfig {
    pub embed_dim: usize,
    pub activation: Activation,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    #[allow(dead_code)]
    pub projection_dim: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L112
#[derive(Clone, Debug)]
struct ClipVisionEmbeddings {
    patch_embedding: Conv2d,
    position_ids: Tensor,
    class_embedding: Tensor,
    position_embedding: quantized_nn::Embedding,
}

impl ClipVisionEmbeddings {
    fn new(vs: VarBuilder, c: &clip::vision_model::ClipVisionConfig) -> Result<Self> {
        // originally nn.Parameter
        let class_embedding = if vs.contains_key("class_embedding") {
            vs.get(c.embed_dim, "class_embedding")?
                .dequantize(vs.device())?
        } else {
            Tensor::randn(0f32, 1f32, c.embed_dim, vs.device())?
        };

        let num_patches = (c.image_size / c.patch_size).pow(2);
        let num_positions = num_patches + 1;
        let position_ids = Tensor::arange(0, num_positions as i64, vs.device())?;

        let conv2dconfig = Conv2dConfig {
            stride: c.patch_size,
            ..Default::default()
        };
        let position_embedding =
            quantized_nn::Embedding::new(num_positions, c.embed_dim, vs.pp("position_embedding"))?;
        let patch_embedding = Conv2d::new(
            vs.get(
                (c.embed_dim, c.num_channels, c.patch_size, c.patch_size),
                "patch_embedding.weight",
            )?
            .dequantize(vs.device())?,
            None,
            conv2dconfig,
        );
        Ok(Self {
            patch_embedding,
            position_ids,
            class_embedding,
            position_embedding,
        })
    }
}

impl Module for ClipVisionEmbeddings {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.shape().dims();
        let patch_embeds = self
            .patch_embedding
            .forward(pixel_values)?
            .flatten_from(2)?
            .transpose(1, 2)?;
        let shape = Shape::from((batch_size[0], 1, self.class_embedding.dim(D::Minus1)?));
        let class_embeds = self.class_embedding.expand(shape)?;
        let embeddings = Tensor::cat(&[class_embeds, patch_embeds], 1)?;
        let position_embedding = self.position_embedding.forward(&self.position_ids)?;
        embeddings.broadcast_add(&position_embedding)
    }
}

// https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L743
#[derive(Clone, Debug)]
pub struct ClipVisionTransformer {
    embeddings: ClipVisionEmbeddings,
    encoder: ClipEncoder,
    pre_layer_norm: candle_nn::LayerNorm,
    final_layer_norm: candle_nn::LayerNorm,
}

impl ClipVisionTransformer {
    pub fn new(vs: VarBuilder, c: &clip::vision_model::ClipVisionConfig) -> Result<Self> {
        let embeddings = ClipVisionEmbeddings::new(vs.pp("embeddings"), c)?;
        let pre_layer_norm = quantized_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("pre_layrnorm"))?;
        let encoder = ClipEncoder::new(vs.pp("encoder"), &clip::EncoderConfig::Vision(c.clone()))?;
        let final_layer_norm =
            quantized_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("post_layernorm"))?;
        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
            pre_layer_norm,
        })
    }
    // required by LLaVA
    pub fn output_hidden_states(&self, pixel_values: &Tensor) -> Result<Vec<Tensor>> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;
        let mut result = self.encoder.output_hidden_states(&hidden_states, None)?;
        let encoder_outputs = result.last().unwrap();
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        result.push(self.final_layer_norm.forward(&pooled_output)?.clone());
        Ok(result)
    }
}

impl Module for ClipVisionTransformer {
    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let hidden_states = pixel_values
            .apply(&self.embeddings)?
            .apply(&self.pre_layer_norm)?;

        let encoder_outputs = self.encoder.forward(&hidden_states, None)?;
        // https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L787
        // pooled_output = encoder_outputs[:, 0, :]
        let pooled_output = encoder_outputs.i((.., 0, ..))?;
        self.final_layer_norm.forward(&pooled_output)
    }
}
