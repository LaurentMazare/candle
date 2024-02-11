#![allow(unused)]
/// A fast implementation of mamba for inference only.
/// This is based on: https://github.com/LaurentMazare/mamba.rs
use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{RmsNorm, VarBuilder};

const D_CONV: usize = 4;
const D_STATE: usize = 16;

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    d_model: usize,
    n_layer: usize,
    vocab_size: usize,
    pad_vocab_size_multiple: usize,
}

impl Config {
    fn vocab_size(&self) -> usize {
        let pad = self.pad_vocab_size_multiple;
        (self.vocab_size + pad - 1) / pad * pad
    }

    fn dt_rank(&self) -> usize {
        (self.d_model + 15) / 16
    }

    fn d_inner(&self) -> usize {
        self.d_model * 2
    }
}

pub struct State {
    hs: Vec<Tensor>,
    prev_xs: Vec<[Tensor; D_CONV]>,
}

impl State {
    pub fn new(batch_size: usize, cfg: &Config, device: &Device) -> Result<Self> {
        let mut hs = Vec::with_capacity(cfg.n_layer);
        let mut prev_xs = Vec::with_capacity(cfg.n_layer);
        for _i in 0..cfg.n_layer {
            let h = Tensor::zeros((batch_size, cfg.d_inner(), D_STATE), DType::F32, device)?;
            let x = Tensor::zeros((batch_size, cfg.d_inner()), DType::F32, device)?;
            hs.push(h);
            prev_xs.push([x.clone(), x.clone(), x.clone(), x.clone()]);
        }
        Ok(Self { hs, prev_xs })
    }
}

#[derive(Clone, Debug)]
pub struct MambaBlock {
    in_proj: Linear,
    conv1d_bias: Tensor,
    conv1d_weights: [Tensor; D_CONV],
    x_proj: Linear,
    dt_proj: Linear,
    a_log: Tensor,
    d: Tensor,
    out_proj: Linear,
    dt_rank: usize,
    layer_index: usize,
}

impl MambaBlock {
    pub fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let d_inner = cfg.d_inner();
        let dt_rank = cfg.dt_rank();
        let in_proj = linear_no_bias(cfg.d_model, d_inner * 2, vb.pp("in_proj"))?;
        let x_proj = linear_no_bias(d_inner, dt_rank + D_STATE * 2, vb.pp("x_proj"))?;
        let dt_proj = linear(dt_rank, d_inner, vb.pp("dt_proj"))?;
        let a_log = vb.get((d_inner, D_STATE), "A_log")?;
        let d = vb.get(d_inner, "D")?;
        let out_proj = linear_no_bias(d_inner, cfg.d_model, vb.pp("out_proj"))?;
        let conv1d_bias = vb.get(d_inner, "conv1d.bias")?;
        let conv1d_weight = vb.get((d_inner, 1, D_CONV), "conv1d.weight")?;
        let conv1d_weights = [
            conv1d_weight.i((.., 0, 0))?,
            conv1d_weight.i((.., 0, 1))?,
            conv1d_weight.i((.., 0, 2))?,
            conv1d_weight.i((.., 0, 3))?,
        ];
        Ok(Self {
            in_proj,
            conv1d_bias,
            conv1d_weights,
            x_proj,
            dt_proj,
            a_log,
            d,
            out_proj,
            dt_rank,
            layer_index,
        })
    }

    pub fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        todo!()
    }
}

#[derive(Clone, Debug)]
pub struct ResidualBlock {
    mixer: MambaBlock,
    norm: RmsNorm,
}

impl ResidualBlock {
    pub fn new(layer_index: usize, cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let norm = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm"))?;
        let mixer = MambaBlock::new(layer_index, cfg, vb.pp("mixer"))?;
        Ok(Self { mixer, norm })
    }

    fn forward(&self, xs: &Tensor, state: &mut State) -> Result<Tensor> {
        self.mixer.forward(&xs.apply(&self.norm)?, state)? + xs
    }
}

// https://github.com/johnma2006/mamba-minimal/blob/61f01953ca153f8c4a850d7111beecbf4be9cee1/model.py#L56
#[derive(Clone, Debug)]
pub struct Model {
    embedding: candle_nn::Embedding,
    layers: Vec<ResidualBlock>,
    norm_f: RmsNorm,
    lm_head: Linear,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size(), cfg.d_model, vb.pp("embedding"))?;
        let mut layers = Vec::with_capacity(cfg.n_layer);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.n_layer {
            let layer = ResidualBlock::new(layer_idx, cfg, vb_l.pp(layer_idx))?;
            layers.push(layer)
        }
        let norm_f = candle_nn::rms_norm(cfg.d_model, 1e-5, vb.pp("norm_f"))?;
        let lm_head = Linear::from_weights(embedding.embeddings().clone(), None);
        Ok(Self {
            embedding,
            layers,
            norm_f,
            lm_head,
        })
    }

    fn forward(&self, input_ids: &Tensor, state: &mut State) -> Result<Tensor> {
        let _b_size = input_ids.dims1()?;
        let mut xs = self.embedding.forward(input_ids)?;
        for layer in self.layers.iter() {
            xs = layer.forward(&xs, state)?
        }
        xs.apply(&self.norm_f)?.apply(&self.lm_head)
    }
}
