//! T5 model implementation.
//!
//! T5 (Text-to-Text Transfer Transformer) is a unified text-to-text transformer model.
//! This implementation follows the original model architecture.
//!
//! Key characteristics:
//! - Text-to-text framework
//! - Relative positional embeddings
//! - T5-specific layer normalization
//! - Encoder-decoder architecture
//! - Support for sequence-to-sequence tasks
//!
//! References:
//! - ⚡ [Interactive Wasm Example](https://huggingface.co/spaces/radames/Candle-T5-Generation-Wasm)
//! - 💻[GH Model](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py)
//! - 🤗 [HF Link](https://huggingface.co/docs/transformers/model_doc/t5)
//! - 📝 [T5 Paper](https://arxiv.org/abs/1910.10683)
//!
//! # Encoder-decoder example:
//!
//! ```bash
//! cargo run --example t5 --release -- \
//!   --model-id "t5-small" \
//!   --prompt "translate to German: A beautiful candle." \
//!   --decode
//! > ...
//! >  Eine schöne Kerze.
//! > 9 tokens generated (2.42 token/s)
//! ```
//!
//! Variants such as [flan-t5](https://huggingface.co/google/flan-t5-small), [flan-ul2](https://huggingface.co/google/flan-ul2) (with `--revision "refs/pr/25"`), and [Co-EdIT](https://huggingface.co/grammarly/coedit-large) are also supported.
//!
//! # Translation with MADLAD
//!
//!
//! [MADLAD-400](https://arxiv.org/abs/2309.04662) is a series of multilingual machine translation T5 models trained on 250 billion tokens covering over 450 languages using publicly available data. These models are competitive with significantly larger models.
//!
//! ```bash
//! cargo run --example t5 --release  -- \
//!   --model-id "jbochi/madlad400-3b-mt" \
//!   --prompt "<2de> How are you, my friend?" \
//!   --decode --temperature 0
//! ...
//!  Wie geht es dir, mein Freund?
//! ```
//!
//! ## Sentence embedding example
//!
//! ```bash
//! cargo run --example t5 --release -- \
//!   --model-id "t5-small" --prompt "A beautiful candle."
//! ...
//! [[[ 0.0515, -0.0541, -0.0761, ..., -0.0392,  0.1511, -0.0265],
//!   [-0.0974,  0.0998, -0.1659, ..., -0.2450,  0.1738, -0.0164],
//!   [ 0.0624, -0.1024,  0.0430, ..., -0.1388,  0.0564, -0.2962],
//!   [-0.0389, -0.1173,  0.0026, ...,  0.1064, -0.1065,  0.0990],
//!   [ 0.1300,  0.0027, -0.0326, ...,  0.0026, -0.0317,  0.0851]]]
//! Tensor[[1, 5, 512], f32]
//! Took 303.766583ms
//! ```

use crate::models::with_tracing::Embedding;
use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Activation, VarBuilder};
use serde::Deserialize;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    span: tracing::Span,
}

pub fn linear_no_bias(d1: usize, d2: usize, vb: VarBuilder) -> Result<Linear> {
    let init_ws = candle_nn::init::DEFAULT_KAIMING_NORMAL;
    let weight = vb.get_with_hints((d2, d1), "weight", init_ws)?;
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    Ok(Linear { weight, span })
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let weight = self.weight.to_dtype(xs.dtype())?;
        let w = match *xs.dims() {
            [b1, b2, _, _] => weight.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => weight.broadcast_left(bsize)?.t()?,
            _ => weight.t()?,
        };
        xs.matmul(&w)
    }
}

fn default_relative_attention_max_distance() -> usize {
    128
}

fn default_is_decoder() -> bool {
    false
}

fn default_use_cache() -> bool {
    true
}

fn default_tie_word_embeddings() -> bool {
    true
}

fn get_mask(size: usize, device: &Device) -> Result<Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    Tensor::from_slice(&mask, (size, size), device)
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[derive(Debug, Deserialize, Default, Clone, PartialEq)]
pub struct ActivationWithOptionalGating {
    pub gated: bool,
    pub activation: candle_nn::Activation,
}

pub fn deserialize_feed_forward_proj_activation<'de, D>(
    deserializer: D,
) -> std::result::Result<ActivationWithOptionalGating, D::Error>
where
    D: serde::de::Deserializer<'de>,
{
    match String::deserialize(deserializer)?.as_str() {
        "gated-gelu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::NewGelu,
        }),
        "gated-silu" => Ok(ActivationWithOptionalGating {
            gated: true,
            activation: candle_nn::Activation::Silu,
        }),
        buf => {
            let activation = serde_plain::from_str(buf).map_err(serde::de::Error::custom)?;
            Ok(ActivationWithOptionalGating {
                gated: false,
                activation,
            })
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    #[serde(default = "default_relative_attention_max_distance")]
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_factor: f64,
    #[serde(default, deserialize_with = "deserialize_feed_forward_proj_activation")]
    pub feed_forward_proj: ActivationWithOptionalGating,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_is_decoder")]
    pub is_decoder: bool,
    pub is_encoder_decoder: bool,
    #[serde(default = "default_use_cache")]
    pub use_cache: bool,
    pub pad_token_id: usize,
    pub eos_token_id: usize,
    pub decoder_start_token_id: Option<usize>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: false,
                activation: Activation::Relu,
            },
            tie_word_embeddings: true,
            is_decoder: false,
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            decoder_start_token_id: Some(0),
        }
    }
}

impl Config {
    // https://huggingface.co/facebook/musicgen-small/blob/495da4ad086b3416a27c6187f9239f9fd96f3962/config.json#L184
    pub fn musicgen_small() -> Self {
        Self {
            d_ff: 3072,
            d_kv: 64,
            d_model: 768,
            dropout_rate: 0.1,
            eos_token_id: 1,
            feed_forward_proj: ActivationWithOptionalGating {
                gated: false,
                activation: Activation::Relu,
            },
            tie_word_embeddings: true,
            initializer_factor: 1.0,
            is_decoder: false,
            is_encoder_decoder: true,
            layer_norm_epsilon: 1e-6,
            num_decoder_layers: Some(12),
            num_heads: 12,
            num_layers: 12,
            pad_token_id: 0,
            decoder_start_token_id: Some(0),
            relative_attention_max_distance: 128,
            relative_attention_num_buckets: 32,
            use_cache: true,
            vocab_size: 32128,
        }
    }
}

#[derive(Debug, Clone)]
struct T5LayerNorm {
    weight: Tensor,
    variance_epsilon: f64,
    span: tracing::Span,
}

impl T5LayerNorm {
    fn load(h: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(h, "weight")?;
        Ok(Self {
            weight,
            variance_epsilon: eps,
            span: tracing::span!(tracing::Level::TRACE, "layer-norm"),
        })
    }
}

impl Module for T5LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let dtype = xs.dtype();
        let xs_f32 = xs.to_dtype(DType::F32)?;
        // variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        let variance = xs_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let xs = xs_f32.broadcast_div(&(variance + self.variance_epsilon)?.sqrt()?)?;
        let xs = xs.to_dtype(dtype)?;
        let xs = xs.broadcast_mul(&self.weight.to_dtype(dtype)?)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseActDense {
    wi: Linear,
    wo: Linear,
    act: Activation,
    span: tracing::Span,
}

impl T5DenseActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi"))?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi,
            wo,
            act: Activation::Relu,
            span: tracing::span!(tracing::Level::TRACE, "dense-act-dense"),
        })
    }
}

impl Module for T5DenseActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let xs = self.wi.forward(xs)?;
        let xs = self.act.forward(&xs)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5DenseGatedActDense {
    wi_0: Linear,
    wi_1: Linear,
    wo: Linear,
    act: Activation,
    span: tracing::Span,
}

impl T5DenseGatedActDense {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let wi_0 = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi_0"))?;
        let wi_1 = linear_no_bias(cfg.d_model, cfg.d_ff, vb.pp("wi_1"))?;
        let wo = linear_no_bias(cfg.d_ff, cfg.d_model, vb.pp("wo"))?;
        Ok(Self {
            wi_0,
            wi_1,
            wo,
            act: cfg.feed_forward_proj.activation,
            span: tracing::span!(tracing::Level::TRACE, "dense-gated-act-dense"),
        })
    }
}

impl Module for T5DenseGatedActDense {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let hidden_gelu = self.act.forward(&self.wi_0.forward(xs)?)?;
        let hidden_linear = self.wi_1.forward(xs)?;
        let xs = hidden_gelu.broadcast_mul(&hidden_linear)?;
        let xs = self.wo.forward(&xs)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5LayerFF {
    dense_act: Option<T5DenseActDense>,
    gated_dense_act: Option<T5DenseGatedActDense>,
    layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5LayerFF {
    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        let (dense_act, gated_dense_act) = if cfg.feed_forward_proj.gated {
            (
                None,
                Some(T5DenseGatedActDense::load(vb.pp("DenseReluDense"), cfg)?),
            )
        } else {
            (
                Some(T5DenseActDense::load(vb.pp("DenseReluDense"), cfg)?),
                None,
            )
        };
        Ok(Self {
            dense_act,
            gated_dense_act,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "layer-ff"),
        })
    }
}

impl Module for T5LayerFF {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let ys = self.layer_norm.forward(xs)?;
        let ys = match &self.dense_act {
            Some(dense_act) => dense_act.forward(&ys)?,
            None => self.gated_dense_act.as_ref().unwrap().forward(&ys)?,
        };
        let xs = (xs + ys)?;
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct T5Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    n_heads: usize,
    d_kv: usize,
    relative_attention_bias: Option<Embedding>,
    relative_attention_num_buckets: usize,
    relative_attention_max_distance: usize,
    inner_dim: usize,
    use_cache: bool,
    kv_cache: Option<(Tensor, Tensor)>,
    span: tracing::Span,
    span_cache: tracing::Span,
    span_mm: tracing::Span,
    span_sm: tracing::Span,
}

impl T5Attention {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &Config,
    ) -> Result<Self> {
        let inner_dim = cfg.num_heads * cfg.d_kv;
        let q = linear_no_bias(cfg.d_model, inner_dim, vb.pp("q"))?;
        let k = linear_no_bias(cfg.d_model, inner_dim, vb.pp("k"))?;
        let v = linear_no_bias(cfg.d_model, inner_dim, vb.pp("v"))?;
        let o = linear_no_bias(inner_dim, cfg.d_model, vb.pp("o"))?;
        let relative_attention_bias = if has_relative_attention_bias {
            let emb = Embedding::new(
                cfg.relative_attention_num_buckets,
                cfg.num_heads,
                vb.pp("relative_attention_bias"),
            )?;
            Some(emb)
        } else {
            None
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            n_heads: cfg.num_heads,
            d_kv: cfg.d_kv,
            relative_attention_bias,
            relative_attention_num_buckets: cfg.relative_attention_num_buckets,
            relative_attention_max_distance: cfg.relative_attention_max_distance,
            inner_dim,
            use_cache: cfg.use_cache && decoder,
            kv_cache: None,
            span: tracing::span!(tracing::Level::TRACE, "attention"),
            span_cache: tracing::span!(tracing::Level::TRACE, "attention-cache"),
            span_mm: tracing::span!(tracing::Level::TRACE, "attention-mm"),
            span_sm: tracing::span!(tracing::Level::TRACE, "attention-sm"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        // Performs Self-attention (if key_value_states is None) or attention
        // over source sentence (provided by key_value_states).
        let _enter = self.span.enter();
        let kv_input = match key_value_states {
            None => xs,
            Some(key_value_states) => key_value_states,
        };
        let (b_sz, q_len) = (xs.dim(0)?, xs.dim(1)?);
        let kv_len = kv_input.dim(1)?;
        let q = self.q.forward(xs)?;
        let k = self.k.forward(kv_input)?;
        let v = self.v.forward(kv_input)?;
        let q = q
            .reshape((b_sz, q_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?
            .contiguous()?;
        let mut k = k
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;
        let mut v = v
            .reshape((b_sz, kv_len, self.n_heads, self.d_kv))?
            .transpose(1, 2)?;

        if self.use_cache && key_value_states.is_none() {
            let _enter = self.span_cache.enter();
            if let Some((kv_cache_k, kv_cache_v)) = &self.kv_cache {
                k = Tensor::cat(&[kv_cache_k, &k], 2)?;
                v = Tensor::cat(&[kv_cache_v, &v], 2)?;
            };
            self.kv_cache = Some((k.clone(), v.clone()));
        };
        let k = k.contiguous()?;
        let v = v.contiguous()?;
        // TODO: Use flash_attn.
        let scores = {
            let _enter = self.span_mm.enter();
            q.matmul(&k.t()?)?
        };
        let scores = match mask {
            None => scores,
            Some(mask) => masked_fill(
                &scores,
                &mask
                    .unsqueeze(0)?
                    .unsqueeze(0)?
                    .repeat((b_sz, self.n_heads))?,
                f32::NEG_INFINITY,
            )?,
        };

        let (scores, position_bias) = match position_bias {
            Some(position_bias) => (
                scores.broadcast_add(position_bias)?,
                Some(position_bias.clone()),
            ),
            None => match &self.relative_attention_bias {
                None => (scores, None),
                Some(relative_attention_bias) => {
                    // This only handles the bidirectional case.
                    let kv_len = k.dim(2)?;
                    let (q_start, q_end) = match self.use_cache {
                        true => ((kv_len - q_len) as u32, kv_len as u32),
                        false => (0_u32, kv_len as u32),
                    };
                    let num_buckets = self.relative_attention_num_buckets as u32 / 2;
                    let max_exact = num_buckets / 2;
                    let relative_position = (q_start..q_end)
                        .map(|i| {
                            (0..kv_len as u32)
                                .map(|j| {
                                    if i < j {
                                        if j - i < max_exact {
                                            j - i + num_buckets
                                        } else {
                                            let b = f32::log(
                                                (j - i) as f32 / max_exact as f32,
                                                self.relative_attention_max_distance as f32
                                                    / max_exact as f32,
                                            ) * (num_buckets - max_exact) as f32;
                                            u32::min(
                                                max_exact + num_buckets + b as u32,
                                                self.relative_attention_num_buckets as u32 - 1,
                                            )
                                        }
                                    } else if i - j < max_exact {
                                        i - j
                                    } else {
                                        let b = f32::log(
                                            (i - j) as f32 / max_exact as f32,
                                            self.relative_attention_max_distance as f32
                                                / max_exact as f32,
                                        ) * (num_buckets - max_exact) as f32;
                                        u32::min(max_exact + b as u32, num_buckets - 1)
                                    }
                                })
                                .collect::<Vec<u32>>()
                        })
                        .collect::<Vec<Vec<_>>>();
                    let relative_buckets = Tensor::new(relative_position, q.device())?;
                    let position_bias = relative_attention_bias
                        .forward(&relative_buckets)?
                        .permute((2, 0, 1))?
                        .unsqueeze(0)?
                        .to_dtype(scores.dtype())?;
                    (scores.broadcast_add(&position_bias)?, Some(position_bias))
                    // TODO: position_bias_masked?
                }
            },
        };

        let attn_weights = {
            let _enter = self.span_sm.enter();
            candle_nn::ops::softmax_last_dim(&scores)?
        };
        let attn_output = attn_weights.matmul(&v)?;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((b_sz, q_len, self.inner_dim))?;
        let attn_output = self.o.forward(&attn_output)?;
        Ok((attn_output, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

#[derive(Debug, Clone)]
struct T5LayerSelfAttention {
    self_attention: T5Attention,
    layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5LayerSelfAttention {
    fn load(h: bool, d: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let self_attention = T5Attention::load(h, d, vb.pp("SelfAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            self_attention,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "self-attn"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let normed_xs = self.layer_norm.forward(xs)?;
        let (ys, position_bias) =
            self.self_attention
                .forward(&normed_xs, position_bias, None, mask)?;
        let ys = (xs + ys)?;
        Ok((ys, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attention.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
struct T5LayerCrossAttention {
    cross_attention: T5Attention,
    layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5LayerCrossAttention {
    fn load(decoder: bool, vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let cross_attention = T5Attention::load(false, decoder, vb.pp("EncDecAttention"), cfg)?;
        let layer_norm =
            T5LayerNorm::load(cfg.d_model, cfg.layer_norm_epsilon, vb.pp("layer_norm"))?;
        Ok(Self {
            cross_attention,
            layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "cross-attn"),
        })
    }

    fn forward(
        &mut self,
        hidden_states: &Tensor,
        position_bias: Option<&Tensor>,
        key_value_states: &Tensor,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        let normed_hidden_states = self.layer_norm.forward(hidden_states)?;
        let (ys, position_bias) = self.cross_attention.forward(
            &normed_hidden_states,
            position_bias,
            Some(key_value_states),
            None,
        )?;
        let ys = (hidden_states + ys)?;
        Ok((ys, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.cross_attention.clear_kv_cache()
    }
}

fn clamp_for_f16(xs: &Tensor) -> Result<Tensor> {
    let mut max = match xs.dtype() {
        DType::U8 => u8::MAX as f64 - 1000.,
        DType::U32 => u32::MAX as f64 - 1000.,
        DType::I64 => i64::MAX as f64 - 1000.,
        DType::F16 => half::f16::MAX.to_f64_const() - 1000.,
        DType::BF16 => half::bf16::MAX.to_f64_const() - 1000.,
        DType::F32 => f32::MAX as f64 - 1000.,
        DType::F64 => f64::MAX - 1000.,
    };
    if xs.is_inf()?.any()? {
        max = max - 1000.;
    }
    xs.clamp(-max, max)
}

#[derive(Debug, Clone)]
struct T5Block {
    self_attn: T5LayerSelfAttention,
    cross_attn: Option<T5LayerCrossAttention>,
    ff: T5LayerFF,
    span: tracing::Span,
}

impl T5Block {
    fn load(
        has_relative_attention_bias: bool,
        decoder: bool,
        vb: VarBuilder,
        cfg: &Config,
    ) -> Result<Self> {
        let vb = vb.pp("layer");
        let self_attn =
            T5LayerSelfAttention::load(has_relative_attention_bias, decoder, vb.pp("0"), cfg)?;
        let cross_attn = if cfg.is_decoder {
            Some(T5LayerCrossAttention::load(decoder, vb.pp("1"), cfg)?)
        } else {
            None
        };
        let ff_i = if cross_attn.is_some() { 2 } else { 1 };
        let ff = T5LayerFF::load(vb.pp(ff_i.to_string()), cfg)?;
        Ok(Self {
            self_attn,
            cross_attn,
            ff,
            span: tracing::span!(tracing::Level::TRACE, "block"),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        position_bias: Option<&Tensor>,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let _enter = self.span.enter();
        // TODO: Cache masks
        let mask = match self.cross_attn.is_some() {
            true => {
                let mask_len = xs.dim(1)?;
                // If the input seq length is 1, no need for a mask, this is also helpful to avoid shape
                // issues when using the KV cache in the decoder.
                if mask_len <= 1 {
                    None
                } else {
                    Some(get_mask(mask_len, xs.device())?)
                }
            }
            false => None,
        };
        let (mut xs, position_bias) = self.self_attn.forward(xs, position_bias, mask.as_ref())?;
        // Clamp for f16
        if xs.dtype() == DType::F16 {
            xs = clamp_for_f16(&xs)?;
        }
        if let Some(cross_attn) = &mut self.cross_attn {
            (xs, _) = cross_attn.forward(&xs, None, encoder_hidden_states.unwrap())?;
            // Clamp for f16
            if xs.dtype() == DType::F16 {
                xs = clamp_for_f16(&xs)?;
            }
        }
        let mut xs = self.ff.forward(&xs)?;
        // Clamp for f16
        if xs.dtype() == DType::F16 {
            xs = clamp_for_f16(&xs)?;
        }
        Ok((xs, position_bias))
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
        self.cross_attn.iter_mut().for_each(|c| c.clear_kv_cache());
    }
}

#[derive(Debug, Clone)]
struct T5Stack {
    block: Vec<T5Block>,
    shared: Arc<Embedding>,
    final_layer_norm: T5LayerNorm,
    span: tracing::Span,
}

impl T5Stack {
    fn load(decoder: bool, vb: VarBuilder, shared: &Arc<Embedding>, cfg: &Config) -> Result<Self> {
        let block = (0..cfg.num_layers)
            .map(|i| T5Block::load(i == 0, decoder, vb.pp(format!("block.{i}")), cfg))
            .collect::<Result<Vec<_>>>()?;
        let final_layer_norm = T5LayerNorm::load(
            cfg.d_model,
            cfg.layer_norm_epsilon,
            vb.pp("final_layer_norm"),
        )?;
        Ok(Self {
            block,
            shared: shared.clone(),
            final_layer_norm,
            span: tracing::span!(tracing::Level::TRACE, "stack"),
        })
    }

    fn forward(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_dt(input_ids, encoder_hidden_states, None)
    }

    fn forward_dt(
        &mut self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        dtype: Option<DType>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let input_embeds = self.shared.as_ref().forward(input_ids)?;
        let input_embeds = match dtype {
            None => input_embeds,
            Some(dtype) => input_embeds.to_dtype(dtype)?,
        };
        let mut hidden_states = input_embeds;
        let mut position_bias = None;
        for block in self.block.iter_mut() {
            (hidden_states, position_bias) = block.forward(
                &hidden_states,
                position_bias.as_ref(),
                encoder_hidden_states,
            )?
        }
        self.final_layer_norm.forward(&hidden_states)
    }

    fn clear_kv_cache(&mut self) {
        self.block.iter_mut().for_each(|b| b.clear_kv_cache())
    }
}

#[derive(Debug, Clone)]
pub struct T5EncoderModel {
    encoder: T5Stack,
    device: Device,
    span: tracing::Span,
}

impl T5EncoderModel {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let shared_vb = if vb.contains_tensor("shared.weight") {
            vb.pp("shared")
        } else if vb.contains_tensor("decoder.embed_tokens") {
            vb.pp("decoder").pp("embed_tokens")
        } else {
            vb.pp("encoder").pp("embed_tokens")
        };
        let shared = Embedding::new(cfg.vocab_size, cfg.d_model, shared_vb)?;
        let shared = Arc::new(shared);
        let encoder = T5Stack::load(false, vb.pp("encoder"), &shared, cfg)?;
        Ok(Self {
            encoder,
            device: vb.device().clone(),
            span: tracing::span!(tracing::Level::TRACE, "encoder"),
        })
    }

    pub fn forward(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.encoder.forward(input_ids, None)
    }

    pub fn forward_dt(&mut self, input_ids: &Tensor, dtype: Option<DType>) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.encoder.forward_dt(input_ids, None, dtype)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn clear_kv_cache(&mut self) {
        self.encoder.clear_kv_cache()
    }
}

#[derive(Debug, Clone)]
pub struct T5ForConditionalGeneration {
    encoder: T5Stack,
    decoder: T5Stack,
    d_model: usize,
    tie_word_embeddings: bool,
    lm_head: Option<Linear>,
    shared: Arc<Embedding>,
    device: Device,
    span_decode: tracing::Span,
    span_decode_head: tracing::Span,
}

impl T5ForConditionalGeneration {
    pub fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        assert!(cfg.is_encoder_decoder);
        let d_model = cfg.d_model;
        let shared_vb = if vb.contains_tensor("shared.weight") {
            vb.pp("shared")
        } else {
            vb.pp("decoder").pp("embed_tokens")
        };
        let shared = Embedding::new(cfg.vocab_size, cfg.d_model, shared_vb)?;
        let shared = Arc::new(shared);

        let mut encoder_cfg = cfg.clone();
        encoder_cfg.is_decoder = false;
        encoder_cfg.use_cache = false;
        encoder_cfg.is_encoder_decoder = false;
        let encoder = T5Stack::load(false, vb.pp("encoder"), &shared, &encoder_cfg)?;

        let mut decoder_cfg = cfg.clone();
        decoder_cfg.is_decoder = true;
        decoder_cfg.is_encoder_decoder = false;
        decoder_cfg.num_layers = cfg.num_decoder_layers.unwrap_or(cfg.num_layers);
        let decoder = T5Stack::load(true, vb.pp("decoder"), &shared, &decoder_cfg)?;

        let tie_word_embeddings = cfg.tie_word_embeddings;
        let lm_head = if tie_word_embeddings {
            None
        } else {
            Some(linear_no_bias(
                cfg.d_model,
                cfg.vocab_size,
                vb.pp("lm_head"),
            )?)
        };

        Ok(Self {
            encoder,
            decoder,
            d_model,
            tie_word_embeddings,
            lm_head,
            shared,
            device: vb.device().clone(),
            span_decode: tracing::span!(tracing::Level::TRACE, "decode"),
            span_decode_head: tracing::span!(tracing::Level::TRACE, "decode-head"),
        })
    }

    pub fn encode(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        self.encoder.forward(input_ids, None)
    }

    pub fn decode(
        &mut self,
        decoder_input_ids: &Tensor,
        encoder_output: &Tensor,
    ) -> Result<Tensor> {
        let _enter = self.span_decode.enter();
        let decoder_output = self
            .decoder
            .forward(decoder_input_ids, Some(encoder_output))?;

        let scaling_factor = if self.tie_word_embeddings {
            // Rescale output before projecting on vocab
            // See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            (self.d_model as f64).sqrt()
        } else {
            1.0
        };
        let sequence_output = ((decoder_output
            .narrow(1, decoder_output.dim(1)? - 1, 1)?
            .squeeze(1)?)
            * scaling_factor)?;
        let output = {
            let _enter = self.span_decode_head.enter();
            match self.lm_head {
                None => sequence_output.matmul(&self.shared.embeddings().t()?)?,
                Some(ref lm_head) => lm_head.forward(&sequence_output)?,
            }
        };
        Ok(output)
    }

    pub fn forward(&mut self, input_ids: &Tensor, decoder_input_ids: &Tensor) -> Result<Tensor> {
        let encoder_output = self.encode(input_ids)?;
        self.decode(decoder_input_ids, &encoder_output)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn clear_kv_cache(&mut self) {
        self.encoder.clear_kv_cache();
        self.decoder.clear_kv_cache();
    }
}
