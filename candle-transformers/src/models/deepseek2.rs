#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{f32::consts::PI, sync::Arc};

use candle::{
    shape::Dim, CpuStorage, CustomOp1, DType, Device, Error, IndexOp, Layout, Result, Shape,
    Tensor, WithDType, D,
};
use candle_nn::{embedding, rms_norm, Activation, Embedding, Linear, Module, RmsNorm, VarBuilder};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Deserialize;

struct NonZero {}

impl NonZero {
    // Sequential version
    fn nonzero<T: WithDType>(&self, vs: &[T], layout: &Layout) -> Vec<u32> {
        let n = layout.dims().len();
        let mut result = Vec::new();
        let mut indices = vec![0u32; n];
        for (i, v) in vs.iter().enumerate() {
            if !v.is_zero() {
                let mut idx = i;
                for (dim_index, dim) in layout.dims().iter().enumerate().rev() {
                    let d = idx % dim;
                    indices[dim_index] = u32::try_from(d).unwrap();
                    idx /= dim;
                }
                result.extend_from_slice(&indices);
            }
        }
        result
    }
}

impl CustomOp1 for NonZero {
    fn name(&self) -> &'static str {
        "nonzero"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            return Err(Error::RequiresContiguous { op: "nonzero" });
        }
        let result = match storage {
            candle::CpuStorage::U8(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::U32(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::I64(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::BF16(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F16(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F32(vs) => self.nonzero(vs, layout),
            candle::CpuStorage::F64(vs) => self.nonzero(vs, layout),
        };
        let index_len = layout.dims().len();
        let result_len = result.len() / index_len;
        let result = CpuStorage::U32(result);
        let shape = Shape::from_dims(&[result_len, index_len]);
        Ok((result, shape))
    }
}

pub trait NonZeroOp {
    fn nonzero(&self) -> Result<Tensor>;
}

impl NonZeroOp for Tensor {
    fn nonzero(&self) -> Result<Tensor> {
        if !self.is_contiguous() {
            return Err(candle::Error::RequiresContiguous { op: "nonzero" });
        }
        let original_device = self.device();
        self.to_device(&candle::Device::Cpu)?
            .apply_op1_no_bwd(&NonZero {})?
            .to_device(original_device)
    }
}

pub struct TopKOutput {
    pub values: Tensor,
    pub indices: Tensor,
}

pub trait TopKLastDimOp {
    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=True.
    fn topk(&self, topk: usize) -> Result<TopKOutput>;

    /// Topk in the last dim. `values` retains a gradient but `indices` has none w.r.t self.
    /// This expects a contiguous tensor.
    /// Note: this implements torch.topk with sorted=False.
    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput>;
}

impl TopKLastDimOp for Tensor {
    fn topk(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let sorted_indices = self.arg_sort_last_dim(false)?;
        let topk_indices = sorted_indices.narrow(D::Minus1, 0, topk)?.contiguous()?;
        Ok(TopKOutput {
            values: self.gather(&topk_indices, D::Minus1)?,
            indices: topk_indices,
        })
    }

    fn topk_unsorted(&self, topk: usize) -> Result<TopKOutput> {
        // Sorted descending
        let sorted_indices_all = self.arg_sort_last_dim(false)?;
        let topk_indices_sorted = sorted_indices_all
            .narrow(D::Minus1, 0, topk)?
            .contiguous()?;
        let topk_values_sorted = self.gather(&topk_indices_sorted, D::Minus1)?;

        // Reorder the indices ascending
        let reorder_indices = topk_indices_sorted.arg_sort_last_dim(true)?;
        let topk_indices_unsorted = topk_indices_sorted.gather(&reorder_indices, D::Minus1)?;
        let topk_values_unsorted = topk_values_sorted.gather(&reorder_indices, D::Minus1)?;
        Ok(TopKOutput {
            values: topk_values_unsorted,
            indices: topk_indices_unsorted,
        })
    }
}

pub trait SplitOp {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>>;
}

impl SplitOp for Tensor {
    fn split<D: Dim>(&self, splits: &[usize], dim: D) -> Result<Vec<Tensor>> {
        let dim = dim.to_index(self.shape(), "split")?;
        let mut split_res = Vec::new();
        let mut index = 0;
        for split in splits {
            split_res.push(self.narrow(dim, index, *split)?);
            index += *split;
        }
        Ok(split_res)
    }
}

pub trait BincountOp {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>>;
}

fn bincount(values: &[u32], minlength: u32) -> Vec<u32> {
    // Find the maximum value in `values` (or zero if empty)
    let max_val = values.par_iter().max().copied().unwrap_or(0);

    // The final size of the bin counts must be at least `minlength`
    // and large enough to include the largest value in `values`.
    let result_len = (max_val + 1).max(minlength);

    // Each thread creates a local histogram (`fold`),
    // and then they are merged together (`reduce`).
    values
        .par_iter()
        .fold(
            // Create a local histogram
            || vec![0u32; result_len as usize],
            // Update the local histogram
            |mut local_counts, &val| {
                local_counts[val as usize] += 1;
                local_counts
            },
        )
        // Merge histograms from all threads
        .reduce(
            // Identity (empty histogram)
            || vec![0u32; result_len as usize],
            // Combine two histograms
            |mut global_counts, local_counts| {
                for (g, l) in global_counts.iter_mut().zip(local_counts) {
                    *g += l;
                }
                global_counts
            },
        )
}

impl BincountOp for Tensor {
    fn bincount(&self, minlength: u32) -> Result<Vec<u32>> {
        let values = self.to_vec1::<u32>()?;

        Ok(bincount(&values, minlength))
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

#[doc(hidden)]
#[macro_export]
macro_rules! serde_default_fn {
    ($t:ty, $name:ident, $v:expr) => {
        fn $name() -> $t {
            $v
        }
    };
}

serde_default_fn!(f64, routed_scaling_factor, 1.0);
serde_default_fn!(TopkMethod, topk_method, TopkMethod::Greedy);
serde_default_fn!(usize, moe_layer_freq, 1);
serde_default_fn!(usize, first_k_dense_replace, 0);
serde_default_fn!(bool, norm_topk_prob, false);
serde_default_fn!(ScoringFunc, scoring_func, ScoringFunc::Softmax);
serde_default_fn!(Activation, hidden_act, Activation::Silu);
serde_default_fn!(bool, tie_word_embeddings, false);

#[derive(Deserialize, Clone, Debug)]
enum TopkMethod {
    #[serde(rename = "greedy")]
    Greedy,
    #[serde(rename = "group_limited_greedy")]
    GroupLimitedGreedy,
}

#[derive(Deserialize, Clone, Debug)]
enum ScoringFunc {
    #[serde(rename = "softmax")]
    Softmax,
}

#[derive(Deserialize, Clone, Debug)]
pub struct DeepSeekV2Config {
    pub(crate) vocab_size: usize,
    pub(crate) hidden_size: usize,
    pub(crate) intermediate_size: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) n_shared_experts: Option<usize>,
    pub(crate) n_routed_experts: Option<usize>,
    #[serde(default = "routed_scaling_factor")]
    pub(crate) routed_scaling_factor: f64,
    #[serde(default = "topk_method")]
    topk_method: TopkMethod,
    pub(crate) num_experts_per_tok: Option<usize>,
    #[serde(default = "moe_layer_freq")]
    pub(crate) moe_layer_freq: usize,
    #[serde(default = "first_k_dense_replace")]
    pub(crate) first_k_dense_replace: usize,
    // k dense layers
    #[serde(default = "norm_topk_prob")]
    pub(crate) norm_topk_prob: bool,
    #[serde(default = "scoring_func")]
    scoring_func: ScoringFunc,
    #[serde(default = "hidden_act")]
    pub(crate) hidden_act: Activation,
    pub(crate) max_position_embeddings: usize,
    pub(crate) rms_norm_eps: f64,
    #[serde(default = "tie_word_embeddings")]
    pub(crate) tie_word_embeddings: bool,
    pub(crate) rope_theta: f32,
    pub(crate) rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub(crate) attention_bias: bool,
    pub(crate) q_lora_rank: Option<usize>,
    pub(crate) qk_rope_head_dim: usize,
    pub(crate) kv_lora_rank: usize,
    pub(crate) v_head_dim: usize,
    pub(crate) qk_nope_head_dim: usize,
    pub(crate) n_group: usize,
    pub(crate) topk_group: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScaledRopeType {
    #[serde(alias = "su")]
    #[serde(alias = "longrope")]
    Su,
    #[serde(alias = "yarn")]
    Yarn,
    #[serde(alias = "dynamic")]
    Dynamic,
    #[serde(alias = "linear")]
    Linear,
}

#[derive(Debug, Clone)]
pub struct DeepSeekV2RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DeepSeekV2RopeScaling {
    Yarn {
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        mscale: f32,
        mscale_all_dim: f32,
        factor: f32,
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
    },
    LinearOrDynamic {
        #[serde(rename = "type")]
        scaling_type: ScaledRopeType,
        factor: f64,
    },
}

pub struct DeepSeekV2RopeConfig {
    pub rope_scaling: Option<DeepSeekV2RopeScaling>,
    pub max_position_embeddings: usize,
    pub rope_theta: f32,
    pub qk_rope_head_dim: usize,
}

impl DeepSeekV2RotaryEmbedding {
    fn new_unscaled(cfg: &DeepSeekV2RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        let max_seq_len = cfg.max_position_embeddings;
        let dim = cfg.qk_rope_head_dim;

        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let sin = freqs.sin()?.to_dtype(dtype)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }

    fn yarn_find_correction_dim(
        num_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> f32 {
        (dim as f32 * (max_position_embeddings as f32 / (num_rot * 2. * PI)).ln())
            / (2. * base.ln())
    }

    fn yarn_find_correction_range(
        low_rot: f32,
        high_rot: f32,
        dim: usize,
        base: f32,
        max_position_embeddings: usize,
    ) -> (f32, f32) {
        let low =
            Self::yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings).floor();
        let high =
            Self::yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings).ceil();
        (low.max(0.), high.min(dim as f32 - 1.))
    }

    fn yarn_linear_ramp_mask(min: f32, mut max: f32, dim: usize, dev: &Device) -> Result<Tensor> {
        if min == max {
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/604d5664dddd88a0433dbae533b7fe9472482de0/modeling_deepseek.py#L255
            max += 0.001;
        }
        let linear_func =
            ((Tensor::arange(0f32, dim as f32, dev)? - min as f64)? / (max as f64 - min as f64))?;
        linear_func.clamp(0., 1.)
    }

    pub(crate) fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
        if scale <= 1. {
            return 1.;
        }
        0.1 * mscale * scale.ln() + 1.
    }

    #[allow(clippy::too_many_arguments)]
    fn new_yarn(
        cfg: &DeepSeekV2RopeConfig,
        dtype: DType,
        dev: &Device,
        original_max_position_embeddings: usize,
        beta_fast: f32,
        beta_slow: f32,
        factor: f32,
        mscale: f32,
        mscale_all_dim: f32,
    ) -> Result<Self> {
        let freq_extra: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32))
            .collect();
        let freq_extra_len = freq_extra.len();
        let freq_extra = Tensor::from_vec(freq_extra, freq_extra_len, dev)?;
        let freq_inter: Vec<_> = (0..cfg.qk_rope_head_dim)
            .step_by(2)
            .map(|i| 1f32 / (factor * cfg.rope_theta.powf(i as f32 / cfg.qk_rope_head_dim as f32)))
            .collect();
        let freq_inter_len = freq_inter.len();
        let freq_inter = Tensor::from_vec(freq_inter, (1, freq_inter_len), dev)?;

        let (low, high) = Self::yarn_find_correction_range(
            beta_fast,
            beta_slow,
            cfg.qk_rope_head_dim,
            cfg.rope_theta,
            original_max_position_embeddings,
        );
        let inv_freq_mask =
            (1. - Self::yarn_linear_ramp_mask(low, high, cfg.qk_rope_head_dim / 2, dev)?)?;
        let inv_freq = freq_inter
            .broadcast_mul(&(1. - &inv_freq_mask)?)?
            .broadcast_add(&freq_extra.broadcast_mul(&inv_freq_mask)?)?;

        let t = Tensor::arange(0u32, cfg.max_position_embeddings as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((cfg.max_position_embeddings, 1))?;
        let freqs = t.matmul(&inv_freq)?;

        let mscale =
            Self::yarn_get_mscale(factor, mscale) / Self::yarn_get_mscale(factor, mscale_all_dim);
        let sin = (freqs.sin()? * mscale as f64)?.to_dtype(dtype)?;
        let cos = (freqs.cos()? * mscale as f64)?.to_dtype(dtype)?;

        Ok(Self { sin, cos })
    }

    pub fn new(cfg: &DeepSeekV2RopeConfig, dtype: DType, dev: &Device) -> Result<Self> {
        match &cfg.rope_scaling {
            Some(DeepSeekV2RopeScaling::LinearOrDynamic {
                scaling_type: _,
                factor: _,
            }) => candle::bail!("linear and dynamic rope are not implemented yet!"),
            Some(DeepSeekV2RopeScaling::Yarn {
                original_max_position_embeddings,
                beta_fast,
                beta_slow,
                factor,
                mscale,
                mscale_all_dim,
                scaling_type: _,
            }) => Self::new_yarn(
                cfg,
                dtype,
                dev,
                *original_max_position_embeddings,
                *beta_fast,
                *beta_slow,
                *factor,
                *mscale,
                *mscale_all_dim,
            ),
            None => Self::new_unscaled(cfg, dtype, dev),
        }
    }

    pub fn forward(
        &self,
        q: &Tensor,
        k: &Tensor,
        seqlen_offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;

        let sin = self.sin.narrow(0, seqlen_offset, seq_len)?;
        let cos = self.cos.narrow(0, seqlen_offset, seq_len)?;

        let q_embed = candle_nn::rotary_emb::rope_i(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope_i(&k.contiguous()?, &cos, &sin)?;

        Ok((q_embed, k_embed))
    }
}

impl DeepSeekV2Config {
    pub(crate) fn q_head_dim(&self) -> usize {
        self.qk_rope_head_dim + self.qk_nope_head_dim
    }

    fn softmax_scale(&self) -> f32 {
        let mut softmax_scale = 1.0 / (self.q_head_dim() as f32).sqrt();
        if let Some(DeepSeekV2RopeScaling::Yarn {
            mscale_all_dim,
            factor,
            ..
        }) = self.rope_scaling
        {
            let mscale = DeepSeekV2RotaryEmbedding::yarn_get_mscale(factor, mscale_all_dim);
            softmax_scale = softmax_scale * mscale * mscale;
        }
        softmax_scale
    }
}

enum QProj {
    Plain(Linear),
    Lora { a: Linear, norm: RmsNorm, b: Linear },
}

impl QProj {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Lora { a, norm, b } => b.forward(&norm.forward(&a.forward(xs)?)?),
            Self::Plain(lin) => lin.forward(xs),
        }
    }
}

struct Attention {
    q: QProj,
    kv_a_proj_with_mqa: Linear,
    kv_a_layernorm: RmsNorm,
    kv_b_proj: Linear,
    o_proj: Linear,
    rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
    cfg: DeepSeekV2Config,
    q_head_dim: usize,
    softmax_scale: f64,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_head_dim = cfg.q_head_dim();
        let q = match cfg.q_lora_rank {
            Some(lora_rank) => {
                let a = candle_nn::linear_b(
                    cfg.hidden_size,
                    lora_rank,
                    cfg.attention_bias,
                    vb.pp("q_a_proj"),
                )?;
                let norm = rms_norm(lora_rank, cfg.rms_norm_eps, vb.pp("q_a_layernorm"))?;
                let b = candle_nn::linear_no_bias(
                    lora_rank,
                    cfg.num_attention_heads * q_head_dim,
                    vb.pp("q_b_proj"),
                )?;
                QProj::Lora { a, norm, b }
            }
            None => QProj::Plain(candle_nn::linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * q_head_dim,
                vb.pp("q_proj"),
            )?),
        };

        let kv_a_proj_with_mqa = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            cfg.attention_bias,
            vb.pp("kv_a_proj_with_mqa"),
        )?;
        let kv_a_layernorm = rms_norm(cfg.kv_lora_rank, cfg.rms_norm_eps, vb.pp("kv_a_layernorm"))?;
        let kv_b_proj = candle_nn::linear_no_bias(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (q_head_dim - cfg.qk_rope_head_dim + cfg.v_head_dim),
            vb.pp("kv_b_proj"),
        )?;

        let o_proj = candle_nn::linear_b(
            cfg.num_attention_heads * cfg.v_head_dim,
            cfg.hidden_size,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;

        Ok(Self {
            q,
            kv_a_proj_with_mqa,
            kv_a_layernorm,
            kv_b_proj,
            o_proj,
            rotary_emb,
            cfg: cfg.clone(),
            q_head_dim,
            softmax_scale: cfg.softmax_scale() as f64,
            kv_cache: None,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let (bs, seq_len, _) = xs.dims3()?;

        let q = {
            let q = self.q.forward(xs)?;
            q.reshape((bs, seq_len, self.cfg.num_attention_heads, self.q_head_dim))?
                .transpose(1, 2)?
        };
        let q_split = q.split(
            &[self.cfg.qk_nope_head_dim, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        let q_nope = q_split[0].clone();
        let q_pe = q_split[1].clone();

        let compressed_kv = self.kv_a_proj_with_mqa.forward(xs)?;
        let ckv_split = compressed_kv.split(
            &[self.cfg.kv_lora_rank, self.cfg.qk_rope_head_dim],
            D::Minus1,
        )?;
        let compressed_kv = ckv_split[0].clone();
        let k_pe = {
            let k_pe = ckv_split[1].clone();
            k_pe.reshape((bs, seq_len, 1, self.cfg.qk_rope_head_dim))?
                .transpose(1, 2)?
        };
        let kv = {
            let kv = self
                .kv_b_proj
                .forward(&self.kv_a_layernorm.forward(&compressed_kv)?)?;
            kv.reshape((
                bs,
                seq_len,
                self.cfg.num_attention_heads,
                self.cfg.qk_nope_head_dim + self.cfg.v_head_dim,
            ))?
            .transpose(1, 2)?
        };

        let kv_split = kv.split(&[self.cfg.qk_nope_head_dim, self.cfg.v_head_dim], D::Minus1)?;
        let k_nope = kv_split[0].clone();
        let v = kv_split[1].clone();

        let (q_pe, k_pe) = self.rotary_emb.forward(&q_pe, &k_pe, seqlen_offset)?;

        let q = Tensor::cat(&[q_nope, q_pe], D::Minus1)?;
        let k = Tensor::cat(&[k_nope, k_pe.repeat((1, q.dim(1)?, 1, 1))?], D::Minus1)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((prev_k, prev_v)) => {
                let key_states = Tensor::cat(&[prev_k, &k], 2)?;
                let value_states = Tensor::cat(&[prev_v, &v], 2)?;
                (key_states, value_states)
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        let attn_out = {
            let att = (q.contiguous()?.matmul(&k.t()?.contiguous()?)? * self.softmax_scale)?;
            let att = match attention_mask {
                Some(mask) => att.broadcast_add(mask)?,
                None => att,
            };

            let att = candle_nn::ops::softmax_last_dim(&att)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?
        };

        let attn_out = if attention_mask.is_some() {
            attn_out.transpose(1, 2)?.reshape((bs, seq_len, ()))?
        } else {
            attn_out.reshape((bs, seq_len, ()))?
        };

        self.o_proj.forward(&attn_out)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache = None
    }
}

struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
    act: Activation,
}

impl Mlp {
    fn new(
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
        hidden_size: Option<usize>,
        intermediate_size: Option<usize>,
    ) -> Result<Self> {
        let hidden_size = hidden_size.unwrap_or(cfg.hidden_size);
        let intermediate_size = intermediate_size.unwrap_or(cfg.intermediate_size);

        Ok(Self {
            gate: candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up: candle_nn::linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down: candle_nn::linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate.forward(xs)?.apply(&self.act)?;
        let rhs = self.up.forward(xs)?;
        self.down.forward(&(&lhs * &rhs)?)
    }
}

struct MoeGate {
    weight: Tensor,
    cfg: DeepSeekV2Config,
    top_k: usize,
    n_routed_experts: usize,
}

impl MoeGate {
    fn new(cfg: &DeepSeekV2Config, vb: VarBuilder, n_routed_experts: usize) -> Result<Self> {
        let weight = vb.get((n_routed_experts, cfg.hidden_size), "weight")?;
        Ok(Self {
            weight,
            cfg: cfg.clone(),
            top_k: cfg.num_experts_per_tok.unwrap(),
            n_routed_experts,
        })
    }

    /// (topk_idx, topk_weight)
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        // Compute gating score
        let xs = xs.reshape(((), h))?;
        let logits = xs
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.weight.t()?.to_dtype(DType::F32)?)?;
        let scores = match self.cfg.scoring_func {
            ScoringFunc::Softmax => candle_nn::ops::softmax_last_dim(&logits)?,
        };

        // Select top-k experts
        let (mut topk_weight, topk_idx) = match self.cfg.topk_method {
            TopkMethod::Greedy => {
                let TopKOutput { values, indices } = scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
            TopkMethod::GroupLimitedGreedy => {
                // (n, n_group)
                let group_scores = scores
                    .reshape((bs * seq_len, self.cfg.n_group, ()))?
                    .max(D::Minus1)?;
                // (n, topk_group)
                let group_idx = scores.topk_unsorted(self.cfg.topk_group)?.indices;
                // (n, n_group)
                let group_mask = group_scores.zeros_like()?.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_scores.dtype())?,
                    1,
                )?;
                // (n, e)
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        bs * seq_len,
                        self.cfg.n_group,
                        self.n_routed_experts / self.cfg.n_group,
                    ))?
                    .reshape((bs, seq_len, ()))?;
                // (n, e)
                // Invert the mask
                let tmp_scores = masked_fill(&score_mask, &(1. - &score_mask.ne(0.)?)?, 0.)?;
                let TopKOutput { values, indices } = tmp_scores.topk_unsorted(self.top_k)?;
                (values, indices)
            }
        };

        if self.top_k > 1 && self.cfg.norm_topk_prob {
            let denominator = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = (topk_weight / denominator)?;
        } else {
            topk_weight = (topk_weight * self.cfg.routed_scaling_factor)?;
        }
        Ok((topk_idx, topk_weight))
    }
}

struct Moe {
    experts: Vec<Mlp>,
    shared_experts: Option<Mlp>,
    gate: MoeGate,
}

impl Moe {
    fn new(
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,

        n_shared_experts: Option<usize>,
        n_routed_experts: usize,
    ) -> Result<Self> {
        let mut experts = Vec::with_capacity(n_routed_experts);
        for i in 0..n_routed_experts {
            let vb_e = vb.pp("experts").pp(i);
            experts.push(Mlp::new(cfg, vb_e, None, Some(cfg.moe_intermediate_size))?);
        }
        let shared_experts = if let Some(n_shared_experts) = n_shared_experts {
            let intermediate_size = cfg.moe_intermediate_size * n_shared_experts;
            Some(Mlp::new(
                cfg,
                vb.pp("shared_experts"),
                None,
                Some(intermediate_size),
            )?)
        } else {
            None
        };
        let gate = MoeGate::new(cfg, vb.pp("gate"), n_routed_experts)?;
        Ok(Self {
            experts,
            shared_experts,
            gate,
        })
    }

    fn moe_infer(&self, xs: &Tensor, topk_ids: &Tensor, topk_weight: &Tensor) -> Result<Tensor> {
        let mut y = xs.zeros_like()?;
        let counts = topk_ids
            .flatten_all()?
            .bincount(self.experts.len() as u32)?;
        for (i, expert) in self.experts.iter().enumerate() {
            if counts[i] == 0 {
                continue;
            }
            let idx_top = topk_ids.eq(i as f64)?.nonzero()?.t()?;
            let idx = &idx_top.i(0)?.contiguous()?;
            let top = &idx_top.i(1)?.contiguous()?;

            y = y.index_add(
                idx,
                &expert.forward(&xs.index_select(idx, 0)?)?.broadcast_mul(
                    &topk_weight
                        .index_select(idx, 0)?
                        .gather(&top.unsqueeze(1)?, 1)?
                        .squeeze(1)?
                        .unsqueeze(D::Minus1)?
                        .to_dtype(xs.dtype())?,
                )?,
                0,
            )?;
        }

        Ok(y)
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let identity = xs.clone();
        let orig_shape = xs.shape();
        let (topk_idx, topk_weight) = self.gate.forward(xs)?;
        let xs = xs.reshape(((), xs.dim(D::Minus1)?))?;

        let mut y = self
            .moe_infer(&xs, &topk_idx, &topk_weight)?
            .reshape(orig_shape)?;
        if let Some(ref shared_experts) = self.shared_experts {
            y = (y + shared_experts.forward(&identity)?)?;
        }
        Ok(y)
    }
}

enum MoeOrMlp {
    Moe(Moe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(mlp) => mlp.forward(xs),
            Self::Moe(moe) => moe.forward(xs),
        }
    }
}

struct DecoderLayer {
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    attn: Attention,
    moe_or_mlp: MoeOrMlp,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<DeepSeekV2RotaryEmbedding>,
        cfg: &DeepSeekV2Config,
        vb: VarBuilder,
        layer_idx: usize,
    ) -> Result<Self> {
        let attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let moe_or_mlp = if cfg.n_routed_experts.is_some()
            && layer_idx >= cfg.first_k_dense_replace
            && layer_idx % cfg.moe_layer_freq == 0
        {
            MoeOrMlp::Moe(Moe::new(
                cfg,
                vb.pp("mlp"),
                cfg.n_shared_experts,
                cfg.n_routed_experts.unwrap(),
            )?)
        } else {
            MoeOrMlp::Mlp(Mlp::new(cfg, vb.pp("mlp"), None, None)?)
        };

        Ok(Self {
            input_layernorm,
            post_attention_layernorm,
            attn,
            moe_or_mlp,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.attn.forward(&xs, attention_mask, seqlen_offset)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .moe_or_mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn clear_kv_cache(&mut self) {
        self.attn.clear_kv_cache();
    }
}

pub struct DeepSeekV2 {
    lm_head: Linear,
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    dtype: DType,
    device: Device,
}

impl DeepSeekV2 {
    pub fn new(cfg: &DeepSeekV2Config, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model");

        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let lm_head = if !cfg.tie_word_embeddings {
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?
        } else {
            candle_nn::Linear::new(embed_tokens.embeddings().clone(), None)
        };
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let rope_cfg = DeepSeekV2RopeConfig {
            rope_scaling: cfg.rope_scaling.clone(),
            max_position_embeddings: cfg.max_position_embeddings,
            rope_theta: cfg.rope_theta,
            qk_rope_head_dim: cfg.qk_rope_head_dim,
        };
        let rotary_emb = Arc::new(DeepSeekV2RotaryEmbedding::new(
            &rope_cfg,
            vb.dtype(),
            vb.device(),
        )?);

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = DecoderLayer::new(rotary_emb.clone(), cfg, vb_l.pp(layer_idx), layer_idx)?;
            layers.push(layer)
        }

        Ok(Self {
            lm_head,
            embed_tokens,
            norm,
            layers,
            dtype: vb.dtype(),
            device: vb.device().clone(),
        })
    }

    fn prepare_decoder_attention_mask(
        &self,
        b_size: usize,
        tgt_len: usize,
        seqlen_offset: usize,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..tgt_len).map(move |j| if i < j { f32::NEG_INFINITY } else { 0. }))
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), &self.device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, &self.device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.expand((b_size, 1, tgt_len, tgt_len + seqlen_offset))?
            .to_dtype(self.dtype)
    }

    pub fn forward(&mut self, input_ids: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (bs, seq_len) = input_ids.dims2()?;
        let mut xs = self.embed_tokens.forward(input_ids)?;
        let attention_mask = if seq_len == 1 {
            None
        } else {
            let mask = self.prepare_decoder_attention_mask(bs, seq_len, seqlen_offset)?;
            Some(mask)
        };
        for layer in &mut self.layers {
            xs = layer.forward(
                &xs,
                attention_mask
                    .as_ref()
                    .map(|m| m.to_device(xs.device()).unwrap())
                    .as_ref(),
                seqlen_offset,
            )?;
        }
        let xs = xs.apply(&self.norm)?;
        let xs = xs.i((.., seq_len - 1, ..))?.contiguous()?;
        let logits = self.lm_head.forward(&xs)?;
        logits.to_dtype(DType::F32)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache();
        }
    }
}
