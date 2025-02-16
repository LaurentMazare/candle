use candle::{Device::Cpu, Result, Tensor};
use candle_nn::{BitLinear, Module};

#[test]
fn test_forward_no_bias() -> Result<()> {
    let weight = Tensor::new(&[[1f32, -1.], [-1., 1.], [1., 1.]], &Cpu)?;
    let layer = BitLinear::new(weight.clone(), None);

    let input = Tensor::new(&[[1f32, -1.]], &Cpu)?;
    let output = layer.forward(&input)?;
    let expected_output = Tensor::new(&[[2.0f32, -2.0, 0.0]], &Cpu)?;

    assert_eq!(output.to_vec2::<f32>()?, expected_output.to_vec2::<f32>()?);
    Ok(())
}
