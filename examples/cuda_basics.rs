#[cfg(feature = "cuda")]
use anyhow::Result;
#[cfg(feature = "cuda")]
use candle::{Device, Tensor};

#[cfg(not(feature = "cuda"))]
fn main() {
    panic!("The examples requires cuda feature");
}

#[cfg(feature = "cuda")]
fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let x = Tensor::new(&[3f32, 1., 4., 1., 5.], &device)?;
    println!("{:?}", x.to_vec1::<f32>()?);
    let y = Tensor::new(&[2f32, 7., 1., 8., 2.], &device)?;
    let z = (y + x * 3.)?;
    println!("{:?}", z.to_vec1::<f32>()?);
    println!("{:?}", z.sqrt()?.to_vec1::<f32>()?);
    let x = Tensor::new(&[[11f32, 22.], [33., 44.], [55., 66.], [77., 78.]], &device)?;
    let y = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &device)?;
    println!("{:?}", y.to_vec2::<f32>()?);
    let z = x.matmul(&y)?;
    println!("{:?}", z.to_vec2::<f32>()?);
    let x = Tensor::new(
        &[[11f32, 22.], [33., 44.], [55., 66.], [77., 78.]],
        &Device::Cpu,
    )?;
    let y = Tensor::new(&[[1f32, 2., 3.], [4., 5., 6.]], &Device::Cpu)?;
    println!("{:?}", y.to_vec2::<f32>()?);
    let z = x.matmul(&y)?;
    println!("{:?}", z.to_vec2::<f32>()?);
    Ok(())
}
