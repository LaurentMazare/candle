//! Loss Calculations
//!
use candle::{Result, Tensor};

/// The negative log likelihood loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///          of categories. This is expected to contain log probabilities.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn nll(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let b_sz = match target.dims() {
        &[b_sz] => b_sz,
        dims => candle::bail!("the target tensor should have a single dimension ({dims:?})"),
    };
    match inp.dims() {
        &[inp_b_sz, _] => {
            if inp_b_sz != b_sz {
                candle::bail!("batch size mismatch between inp ({inp_b_sz}) and target ({b_sz})")
            }
        }
        dims => candle::bail!("the target tensor should have two dimensions ({dims:?})"),
    }
    inp.gather(&target.unsqueeze(1)?, 1)?
        .sum_all()?
        .affine(-1f64 / b_sz as f64, 0.)
}

/// The cross-entropy loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///          of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of u32 of dimension `N`.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn cross_entropy(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    if inp.rank() != 2 {
        candle::bail!("cross_entropy expects an input tensor of rank 2")
    }
    let inp = crate::ops::log_softmax(inp, 1)?;
    nll(&inp, target)
}

/// The mean squared error loss.
pub fn mse(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    (inp - target)?.sqr()?.mean_all()
}

/// The binary cross-entropy with logit loss.
///
/// Arguments
///
/// * [inp]: The input tensor of dimensions `N, C` where `N` is the batch size and `C` the number
///          of categories. This is expected to raw logits.
/// * [target]: The ground truth labels as a tensor of dimension `N, C` where `N` is the batch size and `C` the number
///          of categories.
///
/// The resulting tensor is a scalar containing the average value over the batch.
pub fn binary_cross_entropy_with_logit(inp: &Tensor, target: &Tensor) -> Result<Tensor> {
    let log_sigmoid_input = crate::ops::sigmoid(inp)?.log()?;

    let loss = (1.0 - target)?.mul(inp)?.sub(&log_sigmoid_input)?.mean_all()?;


    Ok(loss)
}
