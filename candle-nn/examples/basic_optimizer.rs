#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use candle::{DType, Device, Result, Tensor};
use candle_nn::{
    linear, AdamW, FnLRScheduler, LRScheduler, Linear, Module, Optimizer, ParamsAdamW, VarBuilder,
    VarMap,
};

fn gen_data() -> Result<(Tensor, Tensor)> {
    // Generate some sample linear data.
    let w_gen = Tensor::new(&[[3f32, 1.]], &Device::Cpu)?;
    let b_gen = Tensor::new(-2f32, &Device::Cpu)?;
    let gen = Linear::new(w_gen, Some(b_gen));
    let sample_xs = Tensor::new(&[[2f32, 1.], [7., 4.], [-4., 12.], [5., 8.]], &Device::Cpu)?;
    let sample_ys = gen.forward(&sample_xs)?;
    Ok((sample_xs, sample_ys))
}

fn main() -> Result<()> {
    let (sample_xs, sample_ys) = gen_data()?;

    // Use backprop to run a linear regression between samples and get the coefficients back.
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
    let model = linear(2, 1, vb.pp("linear"))?;
    let params = ParamsAdamW {
        lr: 0.1,
        ..Default::default()
    };
    let mut opt = AdamW::new(varmap.all_vars(), params)?;
    let mut scheduler = FnLRScheduler::<usize>::new(Box::new(|step| {
        Ok(0.2 * 0.9f64.powi((step as f64 / 1000f64).floor() as i32))
    }));

    for step in 0..10000 {
        opt.set_learning_rate(scheduler.step(step)?);
        let ys = model.forward(&sample_xs)?;
        let loss = ys.sub(&sample_ys)?.sqr()?.sum_all()?;
        opt.backward_step(&loss)?;
        println!("{step} {}", loss.to_vec0::<f32>()?);
    }
    Ok(())
}
