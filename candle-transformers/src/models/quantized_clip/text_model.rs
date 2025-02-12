use candle::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::Module;
use crate::quantized_nn;

use crate::models::clip::text_model::Activation;
use crate::models::clip::{self, EncoderConfig};
use crate::quantized_var_builder::VarBuilder;

// ClipTextEmbeddings mostly based on the existing implementation in the stable diffision model.
// TODO rewrite to be more similar to https://github.com/huggingface/transformers/blob/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip/modeling_clip.py#L142
#[derive(Clone, Debug)]
struct ClipTextEmbeddings {
    token_embedding: quantized_nn::Embedding,
    position_embedding: quantized_nn::Embedding,
    position_ids: Tensor,
}

impl ClipTextEmbeddings {
    fn new(vs: VarBuilder, c: &clip::text_model::ClipTextConfig) -> Result<Self> {
        let token_embedding =
            quantized_nn::Embedding::new(c.vocab_size, c.embed_dim, vs.pp("token_embedding"))?;
        let position_embedding: quantized_nn::Embedding = quantized_nn::Embedding::new(
            c.max_position_embeddings,
            c.embed_dim,
            vs.pp("position_embedding"),
        )?;
        let position_ids =
            Tensor::arange(0u32, c.max_position_embeddings as u32, vs.device())?.unsqueeze(0)?;
        Ok(ClipTextEmbeddings {
            token_embedding,
            position_embedding,
            position_ids,
        })
    }
}

impl Module for ClipTextEmbeddings {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_length = input_ids.dim(D::Minus1)?;
        let inputs_embeds = self.token_embedding.forward(input_ids)?;
        let position_ids = self.position_ids.narrow(1, 0, seq_length)?;
        let position_embedding = self.position_embedding.forward(&position_ids)?;
        inputs_embeds.broadcast_add(&position_embedding)
    }
}

#[derive(Clone, Debug)]
struct ClipAttention {
    k_proj: quantized_nn::Linear,
    v_proj: quantized_nn::Linear,
    q_proj: quantized_nn::Linear,
    out_proj: quantized_nn::Linear,
    head_dim: usize,
    scale: f64,
    num_attention_heads: usize,
}

impl ClipAttention {
    fn new(vs: VarBuilder, c: &clip::EncoderConfig) -> Result<Self> {
        let embed_dim = c.embed_dim();
        let num_attention_heads = c.num_attention_heads();
        let k_proj = quantized_nn::linear(embed_dim, embed_dim, vs.pp("k_proj"))?;
        let v_proj = quantized_nn::linear(embed_dim, embed_dim, vs.pp("v_proj"))?;
        let q_proj = quantized_nn::linear(embed_dim, embed_dim, vs.pp("q_proj"))?;
        let out_proj = quantized_nn::linear(embed_dim, embed_dim, vs.pp("out_proj"))?;
        let head_dim = embed_dim / num_attention_heads;
        let scale = (head_dim as f64).powf(-0.5);

        Ok(ClipAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            head_dim,
            scale,
            num_attention_heads,
        })
    }

    fn shape(&self, xs: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        xs.reshape((bsz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let in_dtype = xs.dtype();
        let (bsz, seq_len, embed_dim) = xs.dims3()?;

        let query_states = (self.q_proj.forward(xs)? * self.scale)?;
        let proj_shape = (bsz * self.num_attention_heads, seq_len, self.head_dim);
        let query_states = self
            .shape(&query_states, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let key_states = self
            .shape(&self.k_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let value_states = self
            .shape(&self.v_proj.forward(xs)?, seq_len, bsz)?
            .reshape(proj_shape)?
            .to_dtype(DType::F32)?;
        let attn_weights = query_states.matmul(&key_states.transpose(1, 2)?)?;

        let src_len = key_states.dim(1)?;

        let attn_weights = if let Some(causal_attention_mask) = causal_attention_mask {
            attn_weights
                .reshape((bsz, self.num_attention_heads, seq_len, src_len))?
                .broadcast_add(causal_attention_mask)?
                .reshape((bsz * self.num_attention_heads, seq_len, src_len))?
        } else {
            attn_weights
        };

        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;

        let attn_output = attn_weights.matmul(&value_states)?.to_dtype(in_dtype)?;
        let attn_output = attn_output
            .reshape((bsz, self.num_attention_heads, seq_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, seq_len, embed_dim))?;
        self.out_proj.forward(&attn_output)
    }
}

#[derive(Clone, Debug)]
struct ClipMlp {
    fc1: quantized_nn::Linear,
    fc2: quantized_nn::Linear,
    activation: Activation,
}

impl ClipMlp {
    fn new(vs: VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let fc1 = quantized_nn::linear(c.embed_dim(), c.intermediate_size(), vs.pp("fc1"))?;
        let fc2 = quantized_nn::linear(c.intermediate_size(), c.embed_dim(), vs.pp("fc2"))?;

        Ok(ClipMlp {
            fc1,
            fc2,
            activation: c.activation(),
        })
    }
}

impl ClipMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.fc1.forward(xs)?;
        self.fc2.forward(&self.activation.forward(&xs)?)
    }
}

#[derive(Clone, Debug)]
struct ClipEncoderLayer {
    self_attn: ClipAttention,
    layer_norm1: candle_nn::LayerNorm,
    mlp: ClipMlp,
    layer_norm2: candle_nn::LayerNorm,
}

impl ClipEncoderLayer {
    fn new(vs: VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let self_attn = ClipAttention::new(vs.pp("self_attn"), c)?;
        let layer_norm1 = quantized_nn::layer_norm(c.embed_dim(), 1e-5, vs.pp("layer_norm1"))?;
        let mlp = ClipMlp::new(vs.pp("mlp"), c)?;
        let layer_norm2 = quantized_nn::layer_norm(c.embed_dim(), 1e-5, vs.pp("layer_norm2"))?;

        Ok(ClipEncoderLayer {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let residual = xs;
        let xs = self.layer_norm1.forward(xs)?;
        let xs = self.self_attn.forward(&xs, causal_attention_mask)?;
        let xs = (xs + residual)?;

        let residual = &xs;
        let xs = self.layer_norm2.forward(&xs)?;
        let xs = self.mlp.forward(&xs)?;
        xs + residual
    }
}

#[derive(Clone, Debug)]
pub struct ClipEncoder {
    layers: Vec<ClipEncoderLayer>,
}

impl ClipEncoder {
    pub fn new(vs: VarBuilder, c: &EncoderConfig) -> Result<Self> {
        let vs = vs.pp("layers");
        let mut layers: Vec<ClipEncoderLayer> = Vec::new();
        for index in 0..c.num_hidden_layers() {
            let layer = ClipEncoderLayer::new(vs.pp(&index.to_string()), c)?;
            layers.push(layer)
        }
        Ok(ClipEncoder { layers })
    }

    pub fn forward(&self, xs: &Tensor, causal_attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut xs = xs.clone();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
        }
        Ok(xs)
    }
    // required by LLaVA
    pub fn output_hidden_states(
        &self,
        xs: &Tensor,
        causal_attention_mask: Option<&Tensor>,
    ) -> Result<Vec<Tensor>> {
        let mut xs = xs.clone();
        let mut hidden_states = Vec::new();
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, causal_attention_mask)?;
            hidden_states.push(xs.clone());
        }
        Ok(hidden_states)
    }
}

/// A CLIP transformer based model.
#[derive(Clone, Debug)]
pub struct ClipTextTransformer {
    embeddings: ClipTextEmbeddings,
    encoder: ClipEncoder,
    final_layer_norm: candle_nn::LayerNorm,
}

impl ClipTextTransformer {
    pub fn new(vs: VarBuilder, c: &clip::text_model::ClipTextConfig) -> Result<Self> {
        let embeddings = ClipTextEmbeddings::new(vs.pp("embeddings"), c)?;
        let encoder = ClipEncoder::new(vs.pp("encoder"), &EncoderConfig::Text(c.clone()))?;
        let final_layer_norm =
            quantized_nn::layer_norm(c.embed_dim, 1e-5, vs.pp("final_layer_norm"))?;
        Ok(ClipTextTransformer {
            embeddings,
            encoder,
            final_layer_norm,
        })
    }

    // TODO: rewrrite to newer version
    fn build_causal_attention_mask(
        bsz: usize,
        seq_len: usize,
        mask_after: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..seq_len)
            .flat_map(|i| {
                (0..seq_len).map(move |j| {
                    if j > i || j > mask_after {
                        f32::MIN
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (seq_len, seq_len), device)?;
        mask.broadcast_as((bsz, 1, seq_len, seq_len))
    }

    pub fn forward_with_mask(&self, input_ids: &Tensor, mask_after: usize) -> Result<Tensor> {
        let (bsz, seq_len) = input_ids.dims2()?;
        let input_ids = self.embeddings.forward(input_ids)?;
        let causal_attention_mask =
            Self::build_causal_attention_mask(bsz, seq_len, mask_after, input_ids.device())?;
        let input_ids = self
            .encoder
            .forward(&input_ids, Some(&causal_attention_mask))?;
        self.final_layer_norm.forward(&input_ids)
    }
}

impl Module for ClipTextTransformer {
    fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let output = self.forward_with_mask(input_ids, usize::MAX)?;
        let sequence_max_indices = input_ids.argmax(D::Minus1)?.to_dtype(DType::I64)?;

        let mut indices = Vec::new();
        for (batch_idx, &seq_idx) in sequence_max_indices.to_vec1::<i64>()?.iter().enumerate() {
            let index = output.i((batch_idx, seq_idx as usize))?.unsqueeze(0)?;
            indices.push(index);
        }
        Tensor::cat(&indices, 0)
    }
}
