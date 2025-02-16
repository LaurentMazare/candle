//! BitLinear layer
//!
//! This layer applies a bit_linear transformation to the incoming data, `y = x@w.t() + b`.
//! The bias is optional. The `forward` method can be used to apply the layer, it supports input
//! with a batch dimension (so of shape `(b_sz, in_c)`) or without (of shape `(in_c,)`), the
//! output has shape `(b_sz, out_c)` and `(out_c,)` respectively.
//!
//! ```rust
//! use candle::{Tensor, Device::Cpu};
//! use candle_nn::{BitLinear, Module};
//! # fn main() -> candle::Result<()> {
//!
//! let w = Tensor::new(&[[1f32, -1.], [-1., 1.], [1., 1.]], &Cpu)?;
//! let layer = BitLinear::new(w, None); // Use no bias.
//! let xs = Tensor::new(&[[1f32, -1.]], &Cpu)?;
//! let ys = layer.forward(&xs)?;
//! assert_eq!(ys.to_vec2::<f32>()?, &[[2.0f32, -2.0, 0.0]]);
//! # Ok(()) }
//! ```
use candle::{Result, Tensor, D};

#[derive(Clone, Debug)]
pub struct BitLinear {
    weight: Tensor,
    bias: Option<Tensor>,
}

fn weight_quant(x: &Tensor) -> Result<Tensor> {
    let scale = (1.0
        / x.to_dtype(candle::DType::F32)?
            .abs()?
            .mean_all()?
            .clamp(1e-5, f32::INFINITY)?)?
    .to_dtype(x.dtype())?;

    let u = (x.broadcast_mul(&scale))?
        .round()?
        .clamp(-1.0, 1.0)?
        .broadcast_div(&scale)?;

    Ok(u)
}

fn activation_quant(x: &Tensor) -> Result<Tensor> {
    let scale = x.abs()?.max_keepdim(D::Minus1)?.clamp(1e-5, f32::INFINITY)?;
    let scale = (127.0 / scale)?;

    let y = (x.broadcast_mul(&scale))?.round()?.clamp(-128., 127.)?.broadcast_div(&scale)?;

    Ok(y)
}

impl BitLinear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        let weight = weight_quant(&weight).unwrap();
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl super::Module for BitLinear {
    fn forward(&self, x: &Tensor) -> candle::Result<Tensor> {
        let w = self.weight();
        let w = match *x.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };

        let x = activation_quant(x)?;

        let x = x.matmul(&w)?;

        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Create or initialize a new bit_linear layer.
///
/// This uses some default names for weights and biases, namely `"weight"` and `"bias"`.
pub fn bit_linear(in_dim: usize, out_dim: usize, vb: crate::VarBuilder) -> Result<BitLinear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_bs = crate::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bs = vb.get_with_hints(out_dim, "bias", init_bs)?;
    Ok(BitLinear::new(ws, Some(bs)))
}

/// Create or initialize a new bit_linear layer without biases.
pub fn bit_linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    vb: crate::VarBuilder,
) -> Result<BitLinear> {
    let init_ws = crate::init::DEFAULT_KAIMING_NORMAL;
    let ws = vb.get_with_hints((out_dim, in_dim), "weight", init_ws)?;
    Ok(BitLinear::new(ws, None))
}

pub fn bit_linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    vb: crate::VarBuilder,
) -> Result<BitLinear> {
    if bias {
        bit_linear(in_dim, out_dim, vb)
    } else {
        bit_linear_no_bias(in_dim, out_dim, vb)
    }
}
