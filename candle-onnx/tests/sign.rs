use candle::{Device, Result, Tensor};
use candle_onnx::onnx::{GraphProto, NodeProto, ValueInfoProto};
use std::collections::HashMap;
mod utils;
#[test]
fn test_sign_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sign".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["X".to_string()],
            output: vec!["Z".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "Z".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert("X".to_string(), Tensor::arange(-2., 3., &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;

    let z = eval.get("Z").expect("Output 'z' not found");
    assert_eq!(
        z.to_dtype(candle::DType::I64)?.to_vec1::<i64>()?.to_vec(),
        vec![-1, -1, 0, 1, 1]
    );

    Ok(())
}
