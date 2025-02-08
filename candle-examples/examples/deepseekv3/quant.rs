use std::rc::Rc;

use candle::{
    quantized::{GgmlDType, QMatMul, QTensor},
    CpuStorage, CustomOp1, DType, Layout, Module, Result, Shape, Tensor,
};
use candle_nn::{var_builder::ShardedVarBuilder, Linear};
use cudarc::nccl::Comm;
use serde::Deserialize;

use super::ops;

#[derive(Debug, Clone, Deserialize)]
pub enum QuantMethodType {
    #[serde(rename = "fp8")]
    Fp8,
}

#[derive(Debug, Clone, Deserialize)]
pub struct QuantizedConfig {
    pub weight_block_size: Option<Vec<usize>>,
    pub quant_method: QuantMethodType,
}

pub struct AllReduce {
    comm: Rc<Comm>,
}

/// This is actually not safe: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/threadsafety.html
/// But for this example purposes, this will work
unsafe impl Sync for AllReduce {}
unsafe impl Send for AllReduce {}

impl CustomOp1 for AllReduce {
    fn name(&self) -> &'static str {
        "allreduce"
    }

    fn cpu_fwd(&self, _s: &CpuStorage, _l: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("AllReduce is never used on cpu")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s: &candle::CudaStorage,
        l: &Layout,
    ) -> Result<(candle::CudaStorage, Shape)> {
        use candle::{backend::BackendStorage, cuda_backend::WrapErr};
        use cudarc::{driver::DeviceSlice, nccl::ReduceOp};
        use half::{bf16, f16};

        let elem_count = l.shape().elem_count();
        let dev = s.device().clone();
        let dst = match s.dtype() {
            DType::BF16 => {
                let s = s.as_cuda_slice::<bf16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<bf16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            DType::F16 => {
                let s = s.as_cuda_slice::<f16>()?;
                let s = match l.contiguous_offsets() {
                    Some((0, l)) if l == s.len() => s,
                    Some(_) | None => candle::bail!("input has to be contiguous"),
                };
                let mut dst = unsafe { dev.alloc::<f16>(elem_count) }.w()?;
                self.comm
                    .all_reduce(s, &mut dst, &ReduceOp::Sum)
                    .map_err(candle::Error::debug)?;
                candle::CudaStorage::wrap_cuda_slice(dst, dev)
            }
            dtype => candle::bail!("unsupported dtype {dtype:?}"),
        };
        Ok((dst, l.shape().clone()))
    }
}

fn shard(dim: usize, rank: usize, world_size: usize) -> candle_nn::var_builder::Shard {
    candle_nn::var_builder::Shard {
        dim,
        rank,
        world_size,
    }
}

/// This linear layer has a weight that is parallelized along the input dimension,
/// returning the "full" output dimension.
pub enum BlockwiseFP8ParallelRowLinear {
    Quantized {
        w: QMatMul,
        b: Option<Tensor>,
        all_reduce: AllReduce,
    },
    Unquantized {
        w: Linear,
        b: Option<Tensor>,
        all_reduce: AllReduce,
    },
}

impl Module for BlockwiseFP8ParallelRowLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized { w, b, all_reduce } => {
                let in_ty = xs.dtype();
                let mut xs = xs.to_dtype(DType::F32)?;

                xs = w.forward(&xs)?.apply_op1_no_bwd(all_reduce)?;
                if let Some(bias) = b {
                    xs = xs.broadcast_add(bias)?;
                }

                xs.to_dtype(in_ty)
            }
            Self::Unquantized { w, b, all_reduce } => {
                let mut xs = w.forward(&xs)?.apply_op1_no_bwd(all_reduce)?;
                if let Some(bias) = b {
                    xs = xs.broadcast_add(bias)?;
                }
                Ok(xs)
            }
        }
    }
}

/// This linear layer has a weight that is parallelized along the output dimension,
/// taking the "full" input dimension.
pub enum BlockwiseFP8ParallelColumnLinear {
    Quantized { w: QMatMul, b: Option<Tensor> },
    Unquantized(Linear),
}

impl Module for BlockwiseFP8ParallelColumnLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized { w, b } => {
                let in_ty = xs.dtype();
                let mut xs = xs.to_dtype(DType::F32)?;

                xs = w.forward(&xs)?;
                if let Some(bias) = b {
                    xs = xs.broadcast_add(bias)?;
                }

                xs.to_dtype(in_ty)
            }
            Self::Unquantized(l) => xs.apply(l),
        }
    }
}

/// This linear layer has a weight that is parallelized along the output dimension,
/// taking the "full" input dimension.
pub enum BlockwiseFP8ReplicatedLinear {
    Quantized { w: QMatMul, b: Option<Tensor> },
    Unquantized(Linear),
}

impl Module for BlockwiseFP8ReplicatedLinear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Quantized { w, b } => {
                let in_ty = xs.dtype();
                let mut xs = xs.to_dtype(DType::F32)?;

                xs = w.forward(&xs)?;
                if let Some(bias) = b {
                    xs = xs.broadcast_add(bias)?;
                }

                xs.to_dtype(in_ty)
            }
            Self::Unquantized(l) => xs.apply(l),
        }
    }
}

/// Load a blockwise quantized FP8 layer and optionally quantize it in-place for faster inference.
/// This linear layer has a weight that is parallelized along the output dimension,
/// taking the "full" input dimension.
///
/// The bias is parallelized.
pub fn blockwise_fp8_linear_b_parallel_column(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    bias: bool,
    quant: Option<GgmlDType>,
    comm: Rc<Comm>,
    vb: ShardedVarBuilder,
) -> Result<BlockwiseFP8ParallelColumnLinear> {
    let rank = comm.rank();
    let size = comm.world_size();

    let Some(config) = config else {
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard(0, rank, size))?;

        let bias = if bias {
            Some(vb.get_with_hints((out_dim,), "bias", shard(0, rank, size))?)
        } else {
            None
        };

        return Ok(BlockwiseFP8ParallelColumnLinear::Unquantized(Linear::new(
            weight, bias,
        )));
    };

    if !matches!(config.quant_method, QuantMethodType::Fp8) {
        candle::bail!("Expected FP8 quant method!");
    }

    let weight_block_size = config
        .weight_block_size
        .as_ref()
        .expect("Blockwise FP8 requires weight_block_size in config");
    if weight_block_size.len() != 2 {
        candle::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        shard(0, rank, size),
        DType::F8E4M3,
    )?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        shard(0, rank, size),
        DType::F32,
    )?;

    let bias_ty = if quant.is_some() {
        DType::F32
    } else {
        vb.dtype()
    };

    let bias = if bias {
        Some(
            vb.get_with_hints((out_dim,), "bias", shard(0, rank, size))?
                .to_dtype(bias_ty)?,
        )
    } else {
        None
    };

    let dequant = ops::fp8_blockwise_dequantize(
        &weight,
        &weight_scale_inv,
        weight_block_size.to_vec(),
        vb.dtype(),
    )?;

    let layer = match quant {
        Some(q) => BlockwiseFP8ParallelColumnLinear::Quantized {
            w: QMatMul::from_qtensor(QTensor::quantize(&dequant, q)?)?,
            b: bias,
        },
        None => BlockwiseFP8ParallelColumnLinear::Unquantized(Linear::new(dequant, None)),
    };

    Ok(layer)
}

/// Load a blockwise quantized FP8 layer and optionally quantize it in-place for faster inference.
/// This linear layer has a weight that is parallelized along the input dimension,
/// returning the "full" output dimension.
///
/// The bias is not parallelized.
pub fn blockwise_fp8_linear_b_parallel_row(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    bias: bool,
    quant: Option<GgmlDType>,
    comm: Rc<Comm>,
    vb: ShardedVarBuilder,
) -> Result<BlockwiseFP8ParallelRowLinear> {
    let rank = comm.rank();
    let size = comm.world_size();

    let all_reduce = AllReduce { comm };

    let Some(config) = config else {
        let weight = vb.get_with_hints((out_dim, in_dim), "weight", shard(0, rank, size))?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        return Ok(BlockwiseFP8ParallelRowLinear::Unquantized {
            w: Linear::new(weight, None),
            b: bias,
            all_reduce,
        });
    };

    if !matches!(config.quant_method, QuantMethodType::Fp8) {
        candle::bail!("Expected FP8 quant method!");
    }

    let weight_block_size = config
        .weight_block_size
        .as_ref()
        .expect("Blockwise FP8 requires weight_block_size in config");
    if weight_block_size.len() != 2 {
        candle::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        shard(1, rank, size),
        DType::F8E4M3,
    )?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        shard(1, rank, size),
        DType::F32,
    )?;

    let bias_ty = if quant.is_some() {
        DType::F32
    } else {
        vb.dtype()
    };

    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?.to_dtype(bias_ty)?)
    } else {
        None
    };

    let dequant = ops::fp8_blockwise_dequantize(
        &weight,
        &weight_scale_inv,
        weight_block_size.to_vec(),
        vb.dtype(),
    )?;

    let layer = match quant {
        Some(q) => BlockwiseFP8ParallelRowLinear::Quantized {
            w: QMatMul::from_qtensor(QTensor::quantize(&dequant, q)?)?,
            b: bias,
            all_reduce,
        },
        None => BlockwiseFP8ParallelRowLinear::Unquantized {
            w: Linear::new(dequant, None),
            b: bias,
            all_reduce,
        },
    };

    Ok(layer)
}

/// Load a blockwise quantized FP8 layer and optionally quantize it in-place for faster inference.
pub fn blockwise_fp8_linear_b_replicated(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    bias: bool,
    quant: Option<GgmlDType>,
    vb: ShardedVarBuilder,
) -> Result<BlockwiseFP8ReplicatedLinear> {
    let Some(config) = config else {
        let weight = vb.get((out_dim, in_dim), "weight")?;

        let bias = if bias {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        return Ok(BlockwiseFP8ReplicatedLinear::Unquantized(Linear::new(
            weight, bias,
        )));
    };

    if !matches!(config.quant_method, QuantMethodType::Fp8) {
        candle::bail!("Expected FP8 quant method!");
    }

    let weight_block_size = config
        .weight_block_size
        .as_ref()
        .expect("Blockwise FP8 requires weight_block_size in config");
    if weight_block_size.len() != 2 {
        candle::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        Default::default(),
        DType::F8E4M3,
    )?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        Default::default(),
        DType::F32,
    )?;

    let bias_ty = if quant.is_some() {
        DType::F32
    } else {
        vb.dtype()
    };

    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?.to_dtype(bias_ty)?)
    } else {
        None
    };

    let dequant = ops::fp8_blockwise_dequantize(
        &weight,
        &weight_scale_inv,
        weight_block_size.to_vec(),
        vb.dtype(),
    )?;

    let layer = match quant {
        Some(q) => BlockwiseFP8ReplicatedLinear::Quantized {
            w: QMatMul::from_qtensor(QTensor::quantize(&dequant, q)?)?,
            b: bias,
        },
        None => BlockwiseFP8ReplicatedLinear::Unquantized(Linear::new(dequant, bias)),
    };

    Ok(layer)
}
