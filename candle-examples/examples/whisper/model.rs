// We use anyhow rather than candle errors as it provides better support for getting the backtrace
// back when using RUST_LIB_BACKTRACE=1.
use anyhow::Result;
use candle::{safetensors::SafeTensors, DType, Device, Shape, Tensor};
use candle_nn::{LayerNorm, Linear};
use serde::Deserialize;
use std::collections::HashMap;

pub struct VarBuilder<'a> {
    safetensors: Option<(HashMap<String, usize>, Vec<SafeTensors<'a>>)>,
    dtype: DType,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    pub fn from_safetensors(
        safetensors: Vec<SafeTensors<'a>>,
        dtype: DType,
        device: &Device,
    ) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }
        Self {
            safetensors: Some((routing, safetensors)),
            device: device.clone(),
            dtype,
        }
    }

    pub fn zeros(dtype: DType, device: Device) -> Self {
        Self {
            safetensors: None,
            device,
            dtype,
        }
    }

    pub fn get<S: Into<Shape>>(&self, s: S, tensor_name: &str) -> candle::Result<Tensor> {
        let s: Shape = s.into();
        match &self.safetensors {
            None => Tensor::zeros(s, self.dtype, &self.device),
            Some((routing, safetensors)) => {
                // Unwrap or 0  just to let the proper error flow.
                let index = routing.get(tensor_name).unwrap_or(&0);
                let tensor = safetensors[*index]
                    .tensor(tensor_name, &self.device)?
                    .to_dtype(self.dtype)?;
                if *tensor.shape() != s {
                    let msg = format!("shape mismatch for {tensor_name}");
                    Err(candle::Error::UnexpectedShape {
                        msg,
                        expected: s,
                        got: tensor.shape().clone(),
                    })?
                }
                Ok(tensor)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HiddenAct {
    Gelu,
    Relu,
}

impl HiddenAct {
    fn forward(&self, xs: &Tensor) -> candle::Result<Tensor> {
        match self {
            Self::Gelu => xs.gelu(),
            Self::Relu => xs.relu(),
        }
    }
}

// The names in comments correspond to the original implementation:
// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L17
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct Config {
    pub num_mel_bins: usize,            // n_mels
    pub max_source_positions: usize,    // n_audio_ctx
    pub d_model: usize,                 // n_audio_state
    pub encoder_attention_heads: usize, // n_audio_head
    pub encoder_layers: usize,          // n_audio_layer
    pub vocab_size: usize,              // n_vocab
    pub max_target_positions: usize,    //  n_text_ctx
    // pub n_text_state: usize,
    pub decoder_attention_heads: usize, // n_text_head
    pub decoder_layers: usize,          // n_text_layer
}

impl Config {
    pub fn tiny_en() -> Self {
        Self {
            num_mel_bins: 80,
            vocab_size: 51864,
            max_source_positions: 1500,
            d_model: 384,
            encoder_attention_heads: 6,
            encoder_layers: 4,
            max_target_positions: 448,
            // n_text_state: 384,
            decoder_attention_heads: 6,
            decoder_layers: 4,
        }
    }
}

struct Embedding {
    embeddings: Tensor,
    hidden_size: usize,
}

impl Embedding {
    fn new(embeddings: Tensor, hidden_size: usize) -> Self {
        Self {
            embeddings,
            hidden_size,
        }
    }

    fn load(vocab_size: usize, hidden_size: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let embeddings = vb.get((vocab_size, hidden_size), &format!("{p}.weight"))?;
        Ok(Self::new(embeddings, hidden_size))
    }

    fn forward(&self, indexes: &Tensor) -> Result<Tensor> {
        let mut final_dims = indexes.dims().to_vec();
        final_dims.push(self.hidden_size);
        let indexes = indexes.flatten_all()?;
        let values = Tensor::embedding(&indexes, &self.embeddings)?;
        let values = values.reshape(final_dims)?;
        Ok(values)
    }
}

fn linear(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
    let bias = vb.get(size2, &format!("{p}.bias"))?;
    Ok(Linear::new(weight, Some(bias)))
}

fn linear_no_bias(size1: usize, size2: usize, p: &str, vb: &VarBuilder) -> Result<Linear> {
    let weight = vb.get((size2, size1), &format!("{p}.weight"))?;
    Ok(Linear::new(weight, None))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ConvConfig {
    padding: usize,
    stride: usize,
}

impl Default for ConvConfig {
    fn default() -> Self {
        Self {
            padding: 0,
            stride: 1,
        }
    }
}

struct Conv1D {
    weight: Tensor,
    bias: Option<Tensor>,
    config: ConvConfig,
}

impl Conv1D {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_channels, in_channels, kernel_size),
            &format!("{p}.weight"),
        )?;
        let bias = vb.get(out_channels, &format!("{p}.bias"))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            config,
        })
    }

    fn load_no_bias(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        config: ConvConfig,
        p: &str,
        vb: &VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get(
            (out_channels, in_channels, kernel_size),
            &format!("{p}.weight"),
        )?;
        Ok(Self {
            weight,
            bias: None,
            config,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.conv1d(&self.weight, self.config.padding, self.config.stride)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => {
                let b = bias.shape().r1()?;
                let bias = bias.reshape((1, b, 1))?;
                Ok(x.broadcast_add(&bias)?)
            }
        }
    }
}

struct Dropout {
    pr: f64,
}

impl Dropout {
    fn new(pr: f64) -> Self {
        Self { pr }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // TODO
        Ok(x.clone())
    }
}

fn layer_norm(size: usize, p: &str, vb: &VarBuilder) -> Result<LayerNorm> {
    let weight = vb.get(size, &format!("{p}.weight"))?;
    let bias = vb.get(size, &format!("{p}.bias"))?;
    Ok(LayerNorm::new(weight, bias, 1e-5))
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L62
struct MultiHeadAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    out: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn load(n_state: usize, n_head: usize, p: &str, vb: &VarBuilder) -> Result<Self> {
        let query = linear(n_state, n_state, &format!("{p}.q_proj"), vb)?;
        let value = linear(n_state, n_state, &format!("{p}.v_proj"), vb)?;
        let key = linear_no_bias(n_state, n_state, &format!("{p}.k_proj"), vb)?;
        let out = linear(n_state, n_state, &format!("{p}.out_proj"), vb)?;
        Ok(Self {
            query,
            key,
            value,
            out,
            n_head,
        })
    }

    fn forward(&self, x: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        let q = self.query.forward(x)?;
        let k = self.key.forward(xa.unwrap_or(x))?;
        let v = self.value.forward(xa.unwrap_or(x))?;
        let wv = self.qkv_attention(&q, &k, &v, mask)?;
        let out = self.out.forward(&wv)?;
        Ok(out)
    }

    fn reshape_head(&self, x: &Tensor) -> Result<Tensor> {
        let (n_batch, n_ctx, n_state) = x.shape().r3()?;
        let target_dims = &[n_batch, n_ctx, self.n_head, n_state / self.n_head];
        Ok(x.reshape(target_dims)?.transpose(1, 2)?)
    }

    fn qkv_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, n_ctx, n_state) = q.shape().r3()?;
        let scale = ((n_state / self.n_head) as f64).powf(-0.25);
        let q = (self.reshape_head(q)? * scale)?;
        let k = (self.reshape_head(k)?.transpose(2, 3)? * scale)?;
        let v = self.reshape_head(v)?.contiguous()?;
        let mut qk = q.matmul(&k)?;
        if let Some(mask) = mask {
            let mask = mask.narrow(0, 0, n_ctx)?.narrow(1, 0, n_ctx)?;
            qk = qk.broadcast_add(&mask)?
        }
        let w = qk.softmax(candle::D::Minus1)?;
        let wv = w.matmul(&v)?.transpose(1, 2)?.flatten_from(2)?;
        Ok(wv)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L111
struct ResidualAttentionBlock {
    attn: MultiHeadAttention,
    attn_ln: LayerNorm,
    cross_attn: Option<(MultiHeadAttention, LayerNorm)>,
    mlp_linear1: Linear,
    mlp_linear2: Linear,
    mlp_ln: LayerNorm,
}

impl ResidualAttentionBlock {
    fn load(n_state: usize, n_head: usize, ca: bool, p: &str, vb: &VarBuilder) -> Result<Self> {
        let attn = MultiHeadAttention::load(n_state, n_head, &format!("{p}.self_attn"), vb)?;
        let attn_ln = layer_norm(n_state, &format!("{p}.self_attn_layer_norm"), vb)?;
        let cross_attn = if ca {
            let cross_attn =
                MultiHeadAttention::load(n_state, n_head, &format!("{p}.encoder_attn"), vb)?;
            let cross_attn_ln = layer_norm(n_state, &format!("{p}.encoder_attn_layer_norm"), vb)?;
            Some((cross_attn, cross_attn_ln))
        } else {
            None
        };
        let n_mlp = n_state * 4;
        let mlp_linear1 = linear(n_state, n_mlp, &format!("{p}.fc1"), vb)?;
        let mlp_linear2 = linear(n_mlp, n_state, &format!("{p}.fc2"), vb)?;
        let mlp_ln = layer_norm(n_state, &format!("{p}.final_layer_norm"), vb)?;
        Ok(Self {
            attn,
            attn_ln,
            cross_attn,
            mlp_linear1,
            mlp_linear2,
            mlp_ln,
        })
    }

    fn forward(&self, x: &Tensor, xa: Option<&Tensor>, mask: Option<&Tensor>) -> Result<Tensor> {
        let attn = self.attn.forward(&self.attn_ln.forward(x)?, None, mask)?;
        let mut x = (x + attn)?;
        if let Some((attn, ln)) = &self.cross_attn {
            x = (&x + attn.forward(&ln.forward(&x)?, xa, None)?)?;
        }
        let mlp = self.mlp_linear2.forward(
            &self
                .mlp_linear1
                .forward(&self.mlp_ln.forward(&x)?)?
                .gelu()?,
        )?;
        Ok((x + mlp)?)
    }
}

fn sinusoids(length: usize, channels: usize) -> Result<Tensor> {
    let max_timescale = 10000f32;
    let log_timescale_increment = max_timescale.ln() / (channels / 2 - 1) as f32;
    let inv_timescales: Vec<_> = (0..channels / 2)
        .map(|i| (i as f32 * (-log_timescale_increment)).exp())
        .collect();
    let arange: Vec<_> = (0..length).map(|c| c as f32).collect();
    let inv_timescales = Tensor::new(inv_timescales.as_slice(), &Device::Cpu)?.unsqueeze(0)?;
    let arange = Tensor::new(arange.as_slice(), &Device::Cpu)?.unsqueeze(1)?;
    let sh = (length, channels / 2);
    let scaled_time = (arange.broadcast_as(sh)? * inv_timescales.broadcast_as(sh)?)?;
    let sincos = Tensor::cat(&[scaled_time.sin()?, scaled_time.cos()?], 1)?;
    Ok(sincos)
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L143
pub struct AudioEncoder {
    conv1: Conv1D,
    conv2: Conv1D,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln_post: LayerNorm,
}

impl AudioEncoder {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.encoder_attention_heads;
        let n_ctx = cfg.max_source_positions;
        let cfg1 = ConvConfig {
            padding: 1,
            stride: 1,
        };
        let cfg2 = ConvConfig {
            padding: 1,
            stride: 2,
        };
        let conv1 = Conv1D::load(
            cfg.num_mel_bins,
            n_state,
            3,
            cfg1,
            &format!("{p}.conv1"),
            vb,
        )?;
        let conv2 = Conv1D::load(n_state, n_state, 3, cfg2, &format!("{p}.conv2"), vb)?;
        let positional_embedding = sinusoids(n_ctx, n_state)?.to_device(&vb.device)?;
        let blocks = (0..cfg.encoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, false, &format!("{p}.layers.{i}"), vb)
            })
            .collect::<Result<Vec<_>>>()?;
        let ln_post = layer_norm(n_state, &format!("{p}.layer_norm"), vb)?;
        Ok(Self {
            conv1,
            conv2,
            positional_embedding,
            blocks,
            ln_post,
        })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(x)?.gelu()?;
        let x = self.conv2.forward(&x)?.gelu()?;
        let x = x.transpose(1, 2)?;
        let (_bsize, seq_len, _hidden) = x.shape().r3()?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, seq_len)?;
        let mut x = x.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, None, None)?
        }
        let x = self.ln_post.forward(&x)?;
        Ok(x)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L176
pub struct TextDecoder {
    token_embedding: Embedding,
    positional_embedding: Tensor,
    blocks: Vec<ResidualAttentionBlock>,
    ln: LayerNorm,
    mask: Tensor,
}

impl TextDecoder {
    fn load(p: &str, vb: &VarBuilder, cfg: &Config) -> Result<Self> {
        let n_state = cfg.d_model;
        let n_head = cfg.decoder_attention_heads;
        let n_ctx = cfg.max_target_positions;
        let token_embedding =
            Embedding::load(cfg.vocab_size, n_state, &format!("{p}.embed_tokens"), vb)?;
        let positional_embedding =
            vb.get((n_ctx, n_state), &format!("{p}.embed_positions.weight"))?;
        let blocks = (0..cfg.decoder_layers)
            .map(|i| {
                ResidualAttentionBlock::load(n_state, n_head, true, &format!("{p}.layers.{i}"), vb)
            })
            .collect::<Result<Vec<_>>>()?;
        let ln = layer_norm(n_state, &format!("{p}.layer_norm"), vb)?;
        let mask: Vec<_> = (0..n_ctx)
            .flat_map(|i| (0..n_ctx).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let mask = Tensor::from_vec(mask, (n_ctx, n_ctx), &vb.device)?;

        Ok(Self {
            token_embedding,
            positional_embedding,
            blocks,
            ln,
            mask,
        })
    }

    pub fn forward(&self, x: &Tensor, xa: &Tensor) -> Result<Tensor> {
        let x_dims = x.dims();
        let last = x_dims[x_dims.len() - 1];
        let token_embedding = self.token_embedding.forward(x)?;
        let positional_embedding = self.positional_embedding.narrow(0, 0, last)?;
        let mut x = token_embedding.broadcast_add(&positional_embedding)?;
        for block in self.blocks.iter() {
            x = block.forward(&x, Some(xa), Some(&self.mask))?;
        }
        let x = self.ln.forward(&x)?;
        let w = self.token_embedding.embeddings.broadcast_left(x_dims[0])?;
        let logits = x.matmul(&w.t()?)?;
        Ok(logits)
    }
}

// https://github.com/openai/whisper/blob/f572f2161ba831bae131364c3bffdead7af6d210/whisper/model.py#L221
pub struct Whisper {
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
    pub config: Config,
}

impl Whisper {
    pub fn load(vb: &VarBuilder, config: Config) -> Result<Self> {
        let encoder = AudioEncoder::load("model.encoder", vb, &config)?;
        let decoder = TextDecoder::load("model.decoder", vb, &config)?;
        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    pub fn forward(&self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let enc = self.encoder.forward(mel)?;
        let dec = self.decoder.forward(tokens, &enc)?;
        Ok(dec)
    }
}
