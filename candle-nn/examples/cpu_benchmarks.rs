/// This example contains some simple benchmarks so that it's easy to run them in perf etc.
#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::quantized::GgmlType;
use candle::{CpuStorage, Device, Layout, Result, Shape, Tensor, D};
use clap::{Parser, Subcommand};

trait Benchmark {
    type PreProcessData;
    type RunResult;

    fn preprocess() -> Result<Self::PreProcessData>;
    fn run_one(_: &Self::PreProcessData) -> Result<Self::RunResult>;

    const ITERS: usize;
}

struct Im2Col(usize, usize);
impl candle::CustomOp1 for Im2Col {
    fn name(&self) -> &'static str {
        "im2col"
    }

    fn cpu_fwd(&self, storage: &CpuStorage, layout: &Layout) -> Result<(CpuStorage, Shape)> {
        let &Self(h_k, w_k) = self;
        let (b, c, h, w) = layout.shape().dims4()?;
        let (h_out, w_out) = (h - h_k + 1, w - w_k + 1);
        let slice = storage.as_slice::<f32>()?;
        let src = match layout.contiguous_offsets() {
            None => candle::bail!("input has to be contiguous"),
            Some((o1, o2)) => &slice[o1..o2],
        };
        let mut dst = vec![0f32; b * h_out * w_out * c * h_k * w_k];
        let (s_b, s_c, s_h) = (c * h * w, h * w, w);
        for b_idx in 0..b {
            let src_idx = b_idx * s_b;
            let dst_idx = b_idx * h_out * w_out * c * h_k * w_k;
            for h_idx in 0..h_out {
                let dst_idx = dst_idx + h_idx * w_out * c * h_k * w_k;
                for w_idx in 0..w_out {
                    let dst_idx = dst_idx + w_idx * c * h_k * w_k;
                    for c_idx in 0..c {
                        let dst_idx = dst_idx + c_idx * h_k * w_k;
                        let src_idx = c_idx * s_c + src_idx;
                        for h_k_idx in 0..h_k {
                            let src_idx = src_idx + (h_idx + h_k_idx) * s_h + w_idx;
                            let dst_idx = dst_idx + h_k_idx * w_k;
                            dst[dst_idx..dst_idx + w_k]
                                .copy_from_slice(&src[src_idx..src_idx + w_k])
                        }
                    }
                }
            }
        }
        let storage = candle::WithDType::to_cpu_storage_owned(dst);
        Ok((storage, (b * h_out * w_out, c * h_k * w_k).into()))
    }
}

// Conv1d example as used in whisper.
struct Conv1d;
impl Benchmark for Conv1d {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let inp = Tensor::randn(0f32, 1., (1, 384, 3000), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1., (384, 384, 3), &Device::Cpu)?;
        Ok((inp, w))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.conv1d(&d.1, 0, 1, 1, 1)
    }

    const ITERS: usize = 5;
}

// Conv2d example as used in stable-diffusion.
struct Conv2d;
impl Benchmark for Conv2d {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;

    fn preprocess() -> Result<Self::PreProcessData> {
        let inp = Tensor::randn(0f32, 1., (2, 320, 96, 96), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1., (320, 320, 3, 3), &Device::Cpu)?;
        Ok((inp, w))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.conv2d(&d.1, 0, 1, 1, 1)
    }

    const ITERS: usize = 5;
}

// Conv2d example as used in stable-diffusion, im2col implementation.
struct Conv2dIm2Col;
impl Benchmark for Conv2dIm2Col {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;

    fn preprocess() -> Result<Self::PreProcessData> {
        let inp = Tensor::randn(0f32, 1., (2, 320, 96, 96), &Device::Cpu)?;
        let w = Tensor::randn(0f32, 1., (320, 320, 3, 3), &Device::Cpu)?;
        Ok((inp, w))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        // d.0.conv2d(&d.1, 0, 1, 1, 1)
        let (b, _, h, w) = d.0.dims4()?;
        let (h_k, w_k) = (3, 3);
        let (h_out, w_out) = (h - h_k + 1, w - w_k + 1);
        let col = d.0.apply_op1_no_bwd(&Im2Col(h_k, w_k))?;
        let res = col.matmul(&d.1.flatten_from(1)?.t()?)?;
        res.reshape((b, (), h_out, w_out))
    }

    const ITERS: usize = 5;
}

struct Matmul;
impl Benchmark for Matmul {
    type PreProcessData = (Tensor, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let lhs = Tensor::randn(0f32, 1., (1024, 1024), &Device::Cpu)?;
        let rhs = Tensor::randn(0f32, 1., (1024, 1024), &Device::Cpu)?;
        Ok((lhs, rhs))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.matmul(&d.1)
    }

    const ITERS: usize = 100;
}

// This benchmark is similar to:
// https://github.com/ggerganov/llama.cpp/blob/master/examples/benchmark/benchmark-matmult.cpp
struct QMatMul;
impl Benchmark for QMatMul {
    type PreProcessData = (candle::quantized::QMatMul, Tensor);
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        let zeros = vec![candle::quantized::k_quants::BlockQ4_0::zeros(); 4096 * 11008 / 32];
        let mm = candle::quantized::QTensor::new(zeros, (4096, 11008))?;
        let mm = candle::quantized::QMatMul::from_qtensor(mm);
        let arg = Tensor::randn(0f32, 1., (128, 11008), &Device::Cpu)?;
        Ok((mm, arg))
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        d.0.forward(&d.1)
    }

    const ITERS: usize = 100;
}

struct Softmax;
impl Benchmark for Softmax {
    type PreProcessData = Tensor;
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        // Typical whisper tiny size.
        let x = Tensor::randn(0f32, 1., (1, 6, 200, 1500), &Device::Cpu)?;
        Ok(x)
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        candle_nn::ops::softmax(d, D::Minus1)
    }

    const ITERS: usize = 100;
}

struct SoftmaxLastDim;
impl Benchmark for SoftmaxLastDim {
    type PreProcessData = Tensor;
    type RunResult = Tensor;
    fn preprocess() -> Result<Self::PreProcessData> {
        // Typical whisper tiny size.
        let x = Tensor::randn(0f32, 1., (1, 6, 200, 1500), &Device::Cpu)?;
        Ok(x)
    }

    fn run_one(d: &Self::PreProcessData) -> Result<Self::RunResult> {
        candle_nn::ops::softmax_last_dim(d)
    }

    const ITERS: usize = 100;
}

fn run<B: Benchmark>(iters: Option<usize>) -> Result<()> {
    use std::hint::black_box;

    let iters = iters.unwrap_or(B::ITERS);
    let d = B::preprocess()?;
    let start = std::time::Instant::now();
    for _iter in 0..iters {
        let _res = black_box(B::run_one(black_box(&d))?);
    }
    println!("{:?}", start.elapsed() / iters as u32);
    Ok(())
}

#[derive(Subcommand, Debug, Clone)]
enum Task {
    Conv1d,
    Conv2d,
    Conv2dIm2Col,
    Matmul,
    Qmatmul,
    Softmax,
    SoftmaxLastDim,
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The benchmark to be run.
    #[command(subcommand)]
    task: Task,

    #[arg(long)]
    iters: Option<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.task {
        Task::Conv1d => run::<Conv1d>(args.iters)?,
        Task::Conv2d => run::<Conv2d>(args.iters)?,
        Task::Conv2dIm2Col => run::<Conv2dIm2Col>(args.iters)?,
        Task::Matmul => run::<Matmul>(args.iters)?,
        Task::Softmax => run::<Softmax>(args.iters)?,
        Task::SoftmaxLastDim => run::<SoftmaxLastDim>(args.iters)?,
        Task::Qmatmul => run::<QMatMul>(args.iters)?,
    }
    Ok(())
}
