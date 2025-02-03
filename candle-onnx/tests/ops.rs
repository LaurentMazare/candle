use candle::test_utils::to_vec2_round;
use candle::{DType, Device, NdArray, Result, Tensor};
use candle_onnx::onnx::attribute_proto::AttributeType;
use candle_onnx::onnx::tensor_proto::DataType;
use candle_onnx::onnx::tensor_shape_proto::{dimension, Dimension};
use candle_onnx::onnx::{type_proto, TensorProto, TensorShapeProto, TypeProto};
use candle_onnx::onnx::{AttributeProto, GraphProto, ModelProto, NodeProto, ValueInfoProto};
use candle_onnx::simple_eval;
use std::collections::HashMap;

mod utils;

const INPUT_X: &str = "x";
const INPUT_Y: &str = "y";
const INPUT_A: &str = "a";
const OUTPUT_Z: &str = "z";

#[test]
fn test_evaluation_fails_without_defined_graph() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(None);
    let inputs: HashMap<String, Tensor> = HashMap::new();
    match candle_onnx::simple_eval(&manual_graph, inputs) {
        Err(err) => assert_eq!(err.to_string(), "no graph defined in proto"),
        Ok(_) => panic!("Expected an error due to undefined graph"),
    }
    Ok(())
}

// "Add"
#[test]
fn test_add_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Add".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 4.0f64);
    Ok(())
}

// "Sub"
#[test]
fn test_sub_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sub".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 0.0f64);
    Ok(())
}

// "Mul"
#[test]
fn test_mul_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Mul".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 4.0f64);
    Ok(())
}

// "Div"
#[test]
fn test_div_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Div".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_vec1::<f64>()?[0];
    assert_eq!(first, 1.0f64);
    Ok(())
}

// "Exp"
#[test]
fn test_exp_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Exp".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(vec![-1.0f32, 0.0f32, 1.0f32, 2.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results[0][0], 0.36787944f32);
    assert_eq!(results[0][1], 1.0f32);
    assert_eq!(results[1], vec![std::f32::consts::E, 7.389056f32]);

    Ok(())
}

// "Equal"
#[test]
fn test_equal_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Equal".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[2.], &Device::Cpu)?);
    inputs.insert(INPUT_Y.to_string(), Tensor::new(&[2.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_dtype(candle::DType::U8)?.to_vec1::<u8>()?.to_vec()[0];
    assert_eq!(first, 1);

    Ok(())
}

// "Not"
#[test]
fn test_not_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Not".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), Tensor::new(&[0.], &Device::Cpu)?);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let first = z.to_dtype(candle::DType::U8)?.to_vec1::<u8>()?.to_vec()[0];
    assert_eq!(first, 1);

    Ok(())
}

// "MatMul"
#[test]
fn test_matmul_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "MatMul".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(
        INPUT_X.to_string(),
        Tensor::from_vec(
            //
            vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );
    inputs.insert(
        INPUT_Y.to_string(),
        Tensor::from_vec(
            //
            vec![5.0f32, 6.0f32, 7.0f32, 8.0f32],
            &[2, 2],
            &Device::Cpu,
        )?,
    );

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let results = z.to_vec2::<f32>()?;
    assert_eq!(results, vec![vec![19.0, 22.0], vec![43.0, 50.0]]);

    Ok(())
}

// "Reshape"
#[test]
fn test_reshape_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Reshape".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;
    let y = Tensor::from_vec(
        //
        vec![4i64],
        &[1],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);
    inputs.insert(INPUT_Y.to_string(), y);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec1::<f32>()?;

    assert_eq!(results, vec![1.0, 2.0, 3.0, 4.0]);

    Ok(())
}

// "LogSoftmax"
#[test]
fn test_logsoftmax_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "LogSoftmax".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.26894143, 0.7310586], vec![0.26894143, 0.7310586]]
    );

    Ok(())
}

// "Softmax"
#[test]
fn test_softmax_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Softmax".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.26894143, 0.7310586], vec![0.26894143, 0.7310586]]
    );

    Ok(())
}

// "Transpose"
#[test]
fn test_transpose_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Transpose".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 3.0], vec![2.0, 4.0]]);

    Ok(())
}

// "Dropout"
#[test]
fn test_dropout_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Dropout".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        //
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    Ok(())
}

// "Flatten"
#[test]
fn test_flatten_operation() -> Result<()> {
    let mut att_axis = AttributeProto {
        name: "axis".to_string(),
        ref_attr_name: "axis".to_string(),
        i: 0,
        doc_string: "axis".to_string(),
        r#type: 2,
        f: 0.0,
        s: vec![],
        t: None,
        g: None,
        sparse_tensor: None,
        tp: None,
        floats: vec![],
        ints: vec![],
        strings: vec![],
        tensors: vec![],
        graphs: vec![],
        sparse_tensors: vec![],
        type_protos: vec![],
    };
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Flatten".to_string(),
            domain: "".to_string(),
            attribute: vec![att_axis.clone()],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![
            1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32, 7.0f32, 8.0f32,
        ],
        &[2, 2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs.clone())?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]);

    att_axis.i = 1;
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Flatten".to_string(),
            domain: "".to_string(),
            attribute: vec![att_axis.clone()],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]
    );

    Ok(())
}

// Below are ops that are implemented but not tested yet

// "MaxPool"
// #[test]

// "AveragePool"
// #[test]

// "BatchNormalization"
// #[test]

// "Squeeze"
// #[test]

// "ConstantOfShape"
#[test]
fn test_constant_of_shape() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
    test(&[4i64, 3, 2], Some(1.), &[1., 1., 1.])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-31
    test(&[0.], Some(0i64), &[0i64])?;

    // "value" defaults to 0 f32
    test(&[1i64, 2, 3, 4], None as Option<i64>, &[0., 0., 0., 0.])?;

    fn test(
        input: impl NdArray,
        value: Option<impl NdArray>,
        expected: impl NdArray,
    ) -> Result<()> {
        let mut attribute = vec![];

        if let Some(value) = value {
            let tensor = Tensor::new(value, &Device::Cpu)?;

            let (value, data_type) = match tensor.dtype() {
                DType::U8 => (
                    tensor.to_vec0::<u8>()?.to_le_bytes().to_vec(),
                    DataType::Uint8,
                ),
                DType::U32 => (
                    tensor.to_vec0::<u32>()?.to_le_bytes().to_vec(),
                    DataType::Uint32,
                ),
                DType::I64 => (
                    tensor.to_vec0::<i64>()?.to_le_bytes().to_vec(),
                    DataType::Int64,
                ),
                DType::F32 => (
                    tensor.to_vec0::<f32>()?.to_le_bytes().to_vec(),
                    DataType::Float,
                ),
                DType::F64 => (
                    tensor.to_vec0::<f64>()?.to_le_bytes().to_vec(),
                    DataType::Double,
                ),
                _ => panic!("unsupported DType in test"),
            };
            let tensor = TensorProto {
                data_type: data_type.into(),
                dims: tensor.dims().iter().map(|v| *v as i64).collect(),
                raw_data: value,
                segment: None,
                float_data: vec![],
                int32_data: vec![],
                string_data: vec![],
                int64_data: vec![],
                name: "".to_string(),
                doc_string: "".to_string(),
                external_data: vec![],
                data_location: 0,
                double_data: vec![],
                uint64_data: vec![],
            };

            attribute.push(AttributeProto {
                name: "value".to_string(),
                ref_attr_name: "value".to_string(),
                i: 0,
                doc_string: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                f: 0.0,
                s: vec![],
                t: Some(tensor),
                g: None,
                sparse_tensor: None,
                tp: None,
                floats: vec![],
                ints: vec![],
                strings: vec![],
                tensors: vec![],
                graphs: vec![],
                sparse_tensors: vec![],
                type_protos: vec![],
            })
        }

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ConstantOfShape".to_string(),
                domain: "".to_string(),
                attribute,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(input, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}

// "Unsqueeze"
#[test]
fn test_unsqueeze() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Unsqueeze".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![
            1.0f32, 2.0f32, //
            3.0f32, 4.0f32, //
        ],
        &[2, 2],
        &Device::Cpu,
    )?;
    let y = Tensor::from_vec(vec![-1i64], &[1], &Device::Cpu)?;

    let inputs = HashMap::from_iter([(INPUT_X.to_string(), x.clone()), (INPUT_Y.to_string(), y)]);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    assert_eq!(z.dims(), &[2, 2, 1]);
    assert_eq!(
        z.flatten_all()?.to_vec1::<f32>()?,
        x.flatten_all()?.to_vec1::<f32>()?
    );

    Ok(())
}

// "Clip"
// #[test]

// "Gather"
#[test]
fn test_gather_operation() -> Result<()> {
    // test taken from https://onnx.ai/onnx/operators/onnx__Gather.html#summary.
    test(
        &[[1.0, 1.2], [2.3, 3.4], [4.5, 5.7]],
        &[[0i64, 1], [1, 2]],
        0,
        &[[[1.0, 1.2], [2.3, 3.4]], [[2.3, 3.4], [4.5, 5.7]]],
    )?;

    // test taken from https://onnx.ai/onnx/operators/onnx__Gather.html#summary.
    test(
        &[[1.0, 1.2, 1.9], [2.3, 3.4, 3.9], [4.5, 5.7, 5.9]],
        &[[0i64, 2]],
        1,
        &[[[1.0, 1.9]], [[2.3, 3.9]], [[4.5, 5.9]]],
    )?;

    // all the tests below are generated from numpy.take, which works like
    // onnx's Gather operation.
    test(&[1.0, 2.0, 3.0, 4.0], 3i64, 0, 4.0)?;

    test(&[[1.0, 2.0, 3.0, 4.0]], 3i64, 1, &[4.0])?;

    test(
        &[[1.0], [2.0], [3.0], [4.0]],
        &[3i64, 2],
        0,
        &[[4.0], [3.0]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        1i64,
        0,
        &[[5.0, 6.0], [7.0, 8.0]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        &[1i64, 0],
        0,
        &[[[5.0, 6.0], [7.0, 8.0]], [[1.0, 2.0], [3.0, 4.0]]],
    )?;

    fn test(
        data: impl NdArray,
        indices: impl NdArray,
        axis: i64,
        expected: impl NdArray,
    ) -> Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis,
            doc_string: "axis".to_string(),
            r#type: 2,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Gather".to_string(),
                domain: "".to_string(),
                attribute: vec![att_axis],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(indices, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}

// GatherElements
#[test]
fn test_gather_elements() -> Result<()> {
    // all the tests below are verified against `torch.gather()`

    // Rank 1 index
    test(&[1.0, 2.0, 3.0, 4.0], &[3i64], 0, &[4.0])?;

    // Rank 2 index
    test(&[[1.0, 2.0, 3.0, 4.0]], &[[3i64]], 1, &[[4.0]])?;
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-57 gather_elements_0
    test(
        &[[1., 2.], [3., 4.]],
        &[[0i64, 0], [1, 0]],
        1,
        &[[1., 1.], [4., 3.]],
    )?;
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-57 gather_elements_1
    test(
        &[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        &[[1i64, 2, 0], [2, 0, 0]],
        0,
        &[[4., 8., 3.], [7., 2., 3.]],
    )?;
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-57 gather_elements_negative_indices
    test(
        &[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
        &[[-1_i64, -2, 0], [-2, 0, 0]],
        0,
        &[[7., 5., 3.], [4., 2., 3.]],
    )?;
    test(
        &[[1.0], [2.0], [3.0], [4.0]],
        &[[3i64], [2]],
        0,
        &[[4.], [3.]],
    )?;

    // Rank 3
    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        &[[[1i64]]],
        0,
        &[[[5.]]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        &[[[1i64]]],
        1,
        &[[[3.]]],
    )?;

    test(
        &[
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
            [[13.0, 14.0], [15.0, 16.0]],
        ],
        &[[[1i64], [0]]],
        2,
        &[[[2.], [3.]]],
    )?;

    // Error cases
    // Invalid index
    assert!(test(&[[1.0, 2.0, 3.0, 4.0]], &[[3i64]], 0, &[[1., 2., 3., 4.]]).is_err());
    // Invalid axis/ dim
    assert!(test(&[[1.0, 2.0, 3.0, 4.0]], &[[3i64]], 2, &[[1., 2., 3., 4.]]).is_err());
    // Invalid rank
    assert!(test(&[[1.0, 2.0, 3.0, 4.0]], &[3i64], 0, &[[1.]]).is_err());

    fn test(
        data: impl NdArray,
        indices: impl NdArray,
        axis: i64,
        expected: impl NdArray,
    ) -> Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis,
            doc_string: "axis".to_string(),
            r#type: 2,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "GatherElements".to_string(),
                domain: "".to_string(),
                attribute: vec![att_axis],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(indices, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Size"
#[test]
fn test_size_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Size".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let results = z.to_scalar::<i64>()?;
    assert_eq!(results, 4);

    Ok(())
}

// "Shape"
#[test]
fn test_shape_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Shape".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    let results = z.to_vec1::<i64>()?;
    assert_eq!(results, vec![2, 2]);

    Ok(())
}

// "Conv"
// #[test]

// "Concat"
// #[test]

// "Abs"
#[test]
fn test_abs_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Abs".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![-1.0f32, 2.0f32, -3.0f32, 4.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![1.0, 2.0], vec![3.0, 4.0]]);

    Ok(())
}

// "Cos"
#[test]
fn test_cos_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Cos".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    assert_eq!(to_vec2_round(z, 4)?, [[1.0, 0.5403], [-0.4161, -0.99]]);
    Ok(())
}

// "Sin"
#[test]
fn test_sin_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sin".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;
    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);
    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);
    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
    assert_eq!(to_vec2_round(z, 4)?, [[0.0, 0.8415], [0.9093, 0.1411]]);
    Ok(())
}

// "Neg"
#[test]
fn test_neg_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Neg".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32, 4.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![-1.0, -2.0], vec![-3.0, -4.0]]);

    Ok(())
}

// "Erf"
// #[test]

// "Tanh"
#[test]
fn test_tanh_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Tanh".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.0, 0.7615942], vec![0.9640276, 0.9950548]]
    );

    Ok(())
}

// "Sigmoid"
#[test]
fn test_sigmoid_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Sigmoid".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.5, 0.7310586], vec![0.880797, 0.95257413]]
    );

    Ok(())
}

// "Gelu"
#[test]
fn test_gelu_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Gelu".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![
            ValueInfoProto {
                name: INPUT_X.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: INPUT_Y.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(vec![0.0f32, 1.0f32, 2.0f32, 3.0f32], &[2, 2], &Device::Cpu)?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(
        results,
        vec![vec![0.0, 0.8413448], vec![1.9544997, 2.9959502]]
    );

    Ok(())
}

// "Relu"
#[test]
fn test_relu_operation() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Relu".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        vec![-1.0f32, 1.0f32, -2.0f32, 3.0f32],
        &[2, 2],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec2::<f32>()?;

    assert_eq!(results, vec![vec![0.0, 1.0], vec![0.0, 3.0]]);

    Ok(())
}

// "Constant"
// #[test]

// "Cast"
// #[test]

// "ReduceMax"
#[test]
fn test_reduce_max() -> Result<()> {
    // Tests with random data generated with `np.random.uniform`
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 bool_inputs
    // No special treatment reqired for bool
    // `np.maximum.reduce(data, axis=axes, keepdims=True)`
    test(
        &[[1_u8, 1], [1, 0], [0, 1], [0, 0]],
        Some(vec![1]),
        1,
        None,
        &[[1_u8], [1], [1], [0]],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 default_axes_keepdims
    // `np.maximum.reduce(data, axis=None, keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        1,
        None,
        &[[[60.]]],
        false,
    )?;
    // same as above but with random
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        1,
        None,
        &[[[9.587318]]],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 default_axes_donot_keep_dims
    // `np.maximum.reduce(data, axis=None, keepdims=False)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        0,
        None,
        60.,
        false,
    )?;
    // same as above but with random
    // `np.maximum.reduce(data, axis=None, keepdims=False)`
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        0,
        None,
        9.587318,
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 keepdims
    // `np.maximum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        1,
        None,
        &[[[20., 2.]], [[40., 2.]], [[60., 2.]]],
        false,
    )?;
    // keepdims with random data
    // `np.maximum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        Some(vec![1]),
        1,
        None,
        &[
            [[-7.318765, 7.2374434]],
            [[6.304022, 4.939862]],
            [[9.587318, 8.008944]],
        ],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-119 negative_axes_keepdims
    // axes = np.array([-1], dtype=np.int64)
    // `np.maximum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1]),
        1,
        None,
        &[[[5.], [20.]], [[30.], [40.]], [[55.], [60.]]],
        false,
    )?;
    // axes = np.array([-2], dtype=np.int64)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2]),
        1,
        None,
        &[[[20., 2.]], [[40., 2.]], [[60., 2.]]],
        false,
    )?;
    // with random
    test(
        &[
            [[-4.1676497, -2.7603748], [-4.5138783, -0.762791]],
            [[-6.3792877, 7.1619177], [-9.958144, 6.3753467]],
            [[9.046973, 3.4554052], [-5.4674335, 5.4642754]],
        ],
        Some(vec![-2]),
        1,
        None,
        &[
            [[-4.1676497, -0.762791]],
            [[-6.3792877, 7.1619177]],
            [[9.046973, 5.4642754]],
        ],
        false,
    )?;

    // Multiple axes - keepdims=1 (true)
    // axes = np.array([0, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 1]),
        1,
        None,
        &[[[60., 2.]]],
        false,
    )?;
    // axes = np.array([0, 2], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 2]),
        1,
        None,
        &[[[55.], [60.]]],
        false,
    )?;
    // axes = np.array([2, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 1]),
        1,
        None,
        &[[[20.]], [[40.]], [[60.]]],
        false,
    )?;
    // axes = np.array([2, 0, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 0, 1]),
        1,
        None,
        &[[[60.]]],
        false,
    )?;
    // Multiple axes - keepdims=0 (false)
    // axes = np.array([0, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 1]),
        0,
        None,
        &[60., 2.],
        false,
    )?;
    // axes = np.array([0, 2], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 2]),
        0,
        None,
        &[55., 60.],
        false,
    )?;
    // axes = np.array([2, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 1]),
        0,
        None,
        &[20., 40., 60.],
        false,
    )?;
    // axes = np.array([2, 0, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 0, 1]),
        0,
        None,
        60.,
        false,
    )?;

    // Multiple axes - negative `axes` - keepdims=1 (true)
    // axes = np.array([-1, 0, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        1,
        None,
        &[[[60.]]],
        false,
    )?;
    // Multiple axes - negative `axes` - keepdims=0 (false)
    // axes = np.array([-1, 0, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        0,
        None,
        60.,
        false,
    )?;

    // `noop_with_empty_axes = true (1)` should yield tensor equivallent to the input tensor
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        0,
        Some(1),
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        false,
    )?;

    // Rank-0 arrays are also valid
    test(42., None, 0, None, 42., false)?;
    test(42., None, 1, None, 42., false)?;

    // Negative test - expect error
    // axes = np.array([-2, 0, 1], dtype=np.int64)
    // np.maximum.reduce(data, axis=tuple(axes), keepdims=True)
    // Should error out with `duplicate value in "axes"`
    assert!(test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2, 0, 1]),
        1,
        None,
        &[[[60.]]],
        false
    )
    .is_err());

    // Negative test - expect error
    // Should error out on empty set
    assert!(test(&[[1_u8; 0]], Some(vec![-2, 0, 1]), 1, None, &[0.], false).is_err());

    // Backward compatibility
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        0,
        None,
        60.,
        true,
    )?;

    fn test(
        data: impl NdArray,
        axes: Option<Vec<i64>>,
        keepdims: i64,
        noop_with_empty_axes: Option<i64>,
        expected: impl NdArray,
        backward_comp: bool,
    ) -> Result<()> {
        let has_axes = axes.is_some();

        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims,
            doc_string: "keepdims".to_string(),
            r#type: 2,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let mut attribute = vec![att_keepdims];
        if let Some(noop) = noop_with_empty_axes {
            if !has_axes {
                let att_no_op_empty_axes = AttributeProto {
                    name: "noop_with_empty_axes".to_string(),
                    ref_attr_name: "noop_with_empty_axes".to_string(),
                    i: noop,
                    doc_string: "noop_with_empty_axes".to_string(),
                    r#type: 2,
                    f: 0.0,
                    s: vec![],
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    tp: None,
                    floats: vec![],
                    ints: vec![],
                    strings: vec![],
                    tensors: vec![],
                    graphs: vec![],
                    sparse_tensors: vec![],
                    type_protos: vec![],
                };

                attribute.push(att_no_op_empty_axes);
            }
        }
        if has_axes && backward_comp {
            attribute.push(AttributeProto {
                name: "axes".to_string(),
                ref_attr_name: "axes".to_string(),
                i: 0,
                doc_string: "axes".to_string(),
                r#type: 7,
                f: 0.0,
                s: vec![],
                t: None,
                g: None,
                sparse_tensor: None,
                tp: None,
                floats: vec![],
                ints: axes.clone().unwrap_or_default(),
                strings: vec![],
                tensors: vec![],
                graphs: vec![],
                sparse_tensors: vec![],
                type_protos: vec![],
            });
        }

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ReduceMax".to_string(),
                domain: "".to_string(),
                attribute,
                input: if has_axes && !backward_comp {
                    vec![INPUT_X.to_string(), INPUT_Y.to_string()]
                } else {
                    vec![INPUT_X.to_string()]
                },
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        let input_tensor = Tensor::new(data, &Device::Cpu)?;
        let input_dtype = input_tensor.dtype();
        inputs.insert(INPUT_X.to_string(), input_tensor);
        if !backward_comp {
            if let Some(a) = axes {
                inputs.insert(INPUT_Y.to_string(), Tensor::new(a, &Device::Cpu)?);
            }
        }

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;

        match expected.dims().len() {
            0 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec0::<u8>()?, expected.to_vec0::<u8>()?)
                } else {
                    assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?)
                }
            }
            1 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec1::<u8>()?, expected.to_vec1::<u8>()?)
                } else {
                    assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?)
                }
            }
            2 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec2::<u8>()?, expected.to_vec2::<u8>()?)
                } else {
                    assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?)
                }
            }
            3 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec3::<u8>()?, expected.to_vec3::<u8>()?)
                } else {
                    assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?)
                }
            }
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}

// "ReduceMin"
#[test]
fn test_reduce_min() -> Result<()> {
    // Tests with random data generated with `np.random.uniform`
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 bool_inputs
    // No special treatment reqired for bool
    // `np.minimum.reduce(data, axis=axes, keepdims=True)`
    test(
        &[[1_u8, 1], [1, 0], [0, 1], [0, 0]],
        Some(vec![1]),
        1,
        None,
        &[[1_u8], [0], [0], [0]],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 default_axes_keepdims
    // `np.minimum.reduce(data, axis=None, keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        1,
        None,
        &[[[1.]]],
        false,
    )?;
    // same as above but with random
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        1,
        None,
        &[[[-8.794852]]],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 default_axes_donot_keep_dims
    // `np.minimum.reduce(data, axis=None, keepdims=False)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        0,
        None,
        1.,
        false,
    )?;
    // same as above but with random
    // `np.minimum.reduce(data, axis=None, keepdims=False)`
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        0,
        None,
        -8.794852,
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 keepdims
    // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        1,
        None,
        &[[[5., 1.]], [[30., 1.]], [[55., 1.]]],
        false,
    )?;
    // keepdims with random data
    // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        Some(vec![1]),
        1,
        None,
        &[
            [[-7.648377, -5.4018507]],
            [[4.5435624, 3.072864]],
            [[-2.5058026, -8.794852]],
        ],
        false,
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-121 negative_axes_keepdims
    // axes = np.array([-1], dtype=np.int64)
    // `np.minimum.reduce(data, axis=tuple(axes), keepdims=True)`
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1]),
        1,
        None,
        &[[[1.], [2.]], [[1.], [2.]], [[1.], [2.]]],
        false,
    )?;
    // axes = np.array([-2], dtype=np.int64)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2]),
        1,
        None,
        &[[[5., 1.]], [[30., 1.]], [[55., 1.]]],
        false,
    )?;
    // with random
    test(
        &[
            [[-4.1676497, -2.7603748], [-4.5138783, -0.762791]],
            [[-6.3792877, 7.1619177], [-9.958144, 6.3753467]],
            [[9.046973, 3.4554052], [-5.4674335, 5.4642754]],
        ],
        Some(vec![-2]),
        1,
        None,
        &[
            [[-4.5138783, -2.7603748]],
            [[-9.958144, 6.3753467]],
            [[-5.4674335, 3.4554052]],
        ],
        false,
    )?;

    // Multiple axes - keepdims=1 (true)
    // axes = np.array([0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 1]),
        1,
        None,
        &[[[5., 1.]]],
        false,
    )?;
    // axes = np.array([0, 2], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 2]),
        1,
        None,
        &[[[1.], [2.]]],
        false,
    )?;
    // axes = np.array([2, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 1]),
        1,
        None,
        &[[[1.]], [[1.]], [[1.]]],
        false,
    )?;
    // axes = np.array([2, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 0, 1]),
        1,
        None,
        &[[[1.]]],
        false,
    )?;
    // Multiple axes - keepdims=0 (false)
    // axes = np.array([0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 1]),
        0,
        None,
        &[5., 1.],
        false,
    )?;
    // axes = np.array([0, 2], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![0, 2]),
        0,
        None,
        &[1., 2.],
        false,
    )?;
    // axes = np.array([2, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 1]),
        0,
        None,
        &[1., 1., 1.],
        false,
    )?;
    // axes = np.array([2, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=False)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![2, 0, 1]),
        0,
        None,
        1.,
        false,
    )?;

    // Multiple axes - negative `axes` - keepdims=1 (true)
    // axes = np.array([-1, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        1,
        None,
        &[[[1.]]],
        false,
    )?;
    // Multiple axes - negative `axes` - keepdims=0 (false)
    // axes = np.array([-1, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        0,
        None,
        1.,
        false,
    )?;

    // `noop_with_empty_axes = true (1)` should yield tensor equivallent to the input tensor
    test(
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        None,
        0,
        Some(1),
        &[
            [[-7.648377, -5.4018507], [-7.318765, 7.2374434]],
            [[6.304022, 4.939862], [4.5435624, 3.072864]],
            [[-2.5058026, 8.008944], [9.587318, -8.794852]],
        ],
        false,
    )?;

    // Rank-0 tensors are also valid
    test(42., None, 0, None, 42., false)?;
    test(42., None, 1, None, 42., false)?;

    // Negative test - expect error
    // axes = np.array([-2, 0, 1], dtype=np.int64)
    // np.minimum.reduce(data, axis=tuple(axes), keepdims=True)
    // Should error out with `duplicate value in "axes"`
    assert!(test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2, 0, 1]),
        1,
        None,
        &[0.],
        false
    )
    .is_err());

    // Negative test - expect error
    // Should error out on empty set
    assert!(test(&[[1_u8; 0]], Some(vec![-2, 0, 1]), 1, None, &[0.], false).is_err());

    // Backward compatibility
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-1, 0, 1]),
        0,
        None,
        1.,
        true,
    )?;

    fn test(
        data: impl NdArray,
        axes: Option<Vec<i64>>,
        keepdims: i64,
        noop_with_empty_axes: Option<i64>,
        expected: impl NdArray,
        backward_comp: bool,
    ) -> Result<()> {
        let has_axes = axes.is_some();

        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims,
            doc_string: "keepdims".to_string(),
            r#type: 2,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let mut attribute = vec![att_keepdims];
        if let Some(noop) = noop_with_empty_axes {
            if !has_axes {
                let att_no_op_empty_axes = AttributeProto {
                    name: "noop_with_empty_axes".to_string(),
                    ref_attr_name: "noop_with_empty_axes".to_string(),
                    i: noop,
                    doc_string: "noop_with_empty_axes".to_string(),
                    r#type: 2,
                    f: 0.0,
                    s: vec![],
                    t: None,
                    g: None,
                    sparse_tensor: None,
                    tp: None,
                    floats: vec![],
                    ints: vec![],
                    strings: vec![],
                    tensors: vec![],
                    graphs: vec![],
                    sparse_tensors: vec![],
                    type_protos: vec![],
                };

                attribute.push(att_no_op_empty_axes);
            }
        }
        if has_axes && backward_comp {
            attribute.push(AttributeProto {
                name: "axes".to_string(),
                ref_attr_name: "axes".to_string(),
                i: 0,
                doc_string: "axes".to_string(),
                r#type: 7,
                f: 0.0,
                s: vec![],
                t: None,
                g: None,
                sparse_tensor: None,
                tp: None,
                floats: vec![],
                ints: axes.clone().unwrap_or_default(),
                strings: vec![],
                tensors: vec![],
                graphs: vec![],
                sparse_tensors: vec![],
                type_protos: vec![],
            });
        }

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ReduceMin".to_string(),
                domain: "".to_string(),
                attribute,
                input: if has_axes && !backward_comp {
                    vec![INPUT_X.to_string(), INPUT_Y.to_string()]
                } else {
                    vec![INPUT_X.to_string()]
                },
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        let input_tensor = Tensor::new(data, &Device::Cpu)?;
        let input_dtype = input_tensor.dtype();
        inputs.insert(INPUT_X.to_string(), input_tensor);
        if !backward_comp {
            if let Some(a) = axes {
                inputs.insert(INPUT_Y.to_string(), Tensor::new(a, &Device::Cpu)?);
            }
        }

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;

        match expected.dims().len() {
            0 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec0::<u8>()?, expected.to_vec0::<u8>()?)
                } else {
                    assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?)
                }
            }
            1 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec1::<u8>()?, expected.to_vec1::<u8>()?)
                } else {
                    assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?)
                }
            }
            2 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec2::<u8>()?, expected.to_vec2::<u8>()?)
                } else {
                    assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?)
                }
            }
            3 => {
                if input_dtype == DType::U8 {
                    assert_eq!(z.to_vec3::<u8>()?, expected.to_vec3::<u8>()?)
                } else {
                    assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?)
                }
            }
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}

// "ReduceMean"
#[test]
fn test_reduce_mean() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 default_axes_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        None,
        1,
        &[[[18.25]]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 do_no_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        0,
        &[[12.5, 1.5], [35.0, 1.5], [57.5, 1.5]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1]),
        1,
        &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-120 negative_axes_keepdims
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![-2]),
        1,
        &[[[12.5, 1.5]], [[35.0, 1.5]], [[57.5, 1.5]]],
    )?;

    // All the test data below was generated based on numpy's np.mean
    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1, 2]),
        0,
        &[7.0, 18.25, 29.5],
    )?;

    test(
        &[
            [[5., 1.], [20., 2.]],
            [[30., 1.], [40., 2.]],
            [[55., 1.], [60., 2.]],
        ],
        Some(vec![1, 2]),
        1,
        &[[[7.0]], [[18.25]], [[29.5]]],
    )?;

    test(&[1., 2., 3.], None, 1, &[2.0])?;

    fn test(
        data: impl NdArray,
        axes: Option<Vec<i64>>,
        keepdims: i64,
        expected: impl NdArray,
    ) -> Result<()> {
        let has_axes = axes.is_some();

        let att_axes = AttributeProto {
            name: "axes".to_string(),
            ref_attr_name: "axes".to_string(),
            i: 0,
            doc_string: "axes".to_string(),
            r#type: 7,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: axes.unwrap_or_default(),
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims,
            doc_string: "keepdims".to_string(),
            r#type: 2,
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };

        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ReduceMean".to_string(),
                domain: "".to_string(),
                attribute: if has_axes {
                    vec![att_axes, att_keepdims]
                } else {
                    vec![att_keepdims]
                },
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Sqrt"
#[test]
fn test_sqrt() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-155
    test(&[1., 4., 9.], &[1., 2., 3.])?;

    fn test(data: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Sqrt".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "RandomUniform"
#[test]
fn test_random_uniform() -> Result<()> {
    test(vec![3, 2, 1, 4], None, None)?;
    test(vec![2, 2, 2, 2], Some(-10.0), None)?;
    test(vec![2, 2, 2, 2], None, Some(10.0))?;
    test(vec![1, 2, 3, 4], Some(-10.0), Some(10.0))?;

    fn test(shape: Vec<i64>, low: Option<f32>, high: Option<f32>) -> Result<()> {
        let att_low = AttributeProto {
            name: "low".to_string(),
            ref_attr_name: "low".to_string(),
            i: 0,
            doc_string: "low".to_string(),
            r#type: 1, // FLOAT
            f: low.unwrap_or(0.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_high = AttributeProto {
            name: "high".to_string(),
            ref_attr_name: "high".to_string(),
            i: 0,
            doc_string: "high".to_string(),
            r#type: 1, // FLOAT
            f: high.unwrap_or(1.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_shape = AttributeProto {
            name: "shape".to_string(),
            ref_attr_name: "shape".to_string(),
            i: 0,
            doc_string: "shape".to_string(),
            r#type: 7, // INTS
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: shape,
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_dtype = AttributeProto {
            name: "dtype".to_string(),
            ref_attr_name: "dtype".to_string(),
            i: 11, // DOUBLE
            doc_string: "dtype".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![att_shape, att_dtype];
            if low.is_some() {
                mut_attrs.push(att_low);
            }
            if high.is_some() {
                mut_attrs.push(att_high);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "RandomUniform".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let eval = candle_onnx::simple_eval(&manual_graph, HashMap::new())?;
        assert_eq!(eval.len(), 1);
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
        let min = z
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .reduce(f64::min)
            .unwrap();
        let max = z
            .flatten_all()?
            .to_vec1()?
            .into_iter()
            .reduce(f64::max)
            .unwrap();
        assert!(min >= low.unwrap_or(0.0).into());
        assert!(max <= high.unwrap_or(1.0).into());
        assert_ne!(min, max);
        Ok(())
    }

    Ok(())
}

// "RandomNormal"
#[test]
fn test_random_normal() -> Result<()> {
    test(vec![3, 2, 1, 4], None, None)?;
    test(vec![2, 2, 2, 2], Some(-10.0), None)?;
    test(vec![2, 2, 2, 2], None, Some(10.0))?;
    test(vec![1, 2, 3, 4], Some(-10.0), Some(10.0))?;

    fn test(shape: Vec<i64>, mean: Option<f32>, scale: Option<f32>) -> Result<()> {
        let att_mean = AttributeProto {
            name: "mean".to_string(),
            ref_attr_name: "mean".to_string(),
            i: 0,
            doc_string: "mean".to_string(),
            r#type: 1, // FLOAT
            f: mean.unwrap_or(0.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_scale = AttributeProto {
            name: "scale".to_string(),
            ref_attr_name: "scale".to_string(),
            i: 0,
            doc_string: "scale".to_string(),
            r#type: 1, // FLOAT
            f: scale.unwrap_or(1.0),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_shape = AttributeProto {
            name: "shape".to_string(),
            ref_attr_name: "shape".to_string(),
            i: 0,
            doc_string: "shape".to_string(),
            r#type: 7, // INTS
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: shape,
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_dtype = AttributeProto {
            name: "dtype".to_string(),
            ref_attr_name: "dtype".to_string(),
            i: 11, // DOUBLE
            doc_string: "dtype".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![att_shape, att_dtype];
            if mean.is_some() {
                mut_attrs.push(att_mean);
            }
            if scale.is_some() {
                mut_attrs.push(att_scale);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "RandomNormal".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let eval = candle_onnx::simple_eval(&manual_graph, HashMap::new())?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");
        let data = z.flatten_all()?.to_vec1::<f64>()?;

        // test if values are unique
        for (i, a) in data.iter().enumerate() {
            for (j, b) in data.iter().enumerate() {
                if i == j {
                    continue;
                };
                assert_ne!(a, b);
            }
        }

        Ok(())
    }

    Ok(())
}

// "Range"
#[test]
fn test_range() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
    test(1., 5., 2., &[1., 3.])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-113
    test(10i64, 6i64, -3i64, &[10i64, 7i64])?;

    fn test(
        start: impl NdArray,
        limit: impl NdArray,
        delta: impl NdArray,
        expected: impl NdArray,
    ) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Range".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    INPUT_X.to_string(),
                    INPUT_Y.to_string(),
                    INPUT_A.to_string(),
                ],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(start, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(limit, &Device::Cpu)?);
        inputs.insert(INPUT_A.to_string(), Tensor::new(delta, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Greater"
#[test]
fn test_greater() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
    test(&[1., 2., 3.], &[3., 2., 1.], &[0u8, 0, 1])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-63
    test(&[1., 2., 3.], 2., &[0u8, 0, 1])?;

    fn test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Greater".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(b, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Less"
#[test]
fn test_less() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
    test(&[1., 2., 3.], &[3., 2., 1.], &[1u8, 0, 0])?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-81
    test(&[1., 2., 3.], 2., &[1u8, 0, 0])?;

    fn test(a: impl NdArray, b: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Less".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(b, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Log"
#[test]
fn test_log() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-82
    test(&[1., 10.], &[0., std::f64::consts::LN_10])?;

    fn test(data: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Log".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Min"
#[test]
fn test_min() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-94
    test(&[3., 2., 1.], &[1., 4., 4.], &[2., 5., 0.], &[1., 2., 0.])?;

    fn test(
        a: impl NdArray,
        b: impl NdArray,
        c: impl NdArray,
        expected: impl NdArray,
    ) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Min".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    INPUT_X.to_string(),
                    INPUT_Y.to_string(),
                    INPUT_A.to_string(),
                ],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(a, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(b, &Device::Cpu)?);
        inputs.insert(INPUT_A.to_string(), Tensor::new(c, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "Where"
#[test]
fn test_where() -> Result<()> {
    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
    test(
        &[[1u8, 0], [1, 1]],
        &[[1i64, 2], [3, 4]],
        &[[9i64, 8], [7, 6]],
        &[[1i64, 8], [3, 4]],
    )?;

    // https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-173
    test(
        &[[1u8, 0], [1, 1]],
        &[[1., 2.], [3., 4.]],
        &[[9., 8.], [7., 6.]],
        &[[1., 8.], [3., 4.]],
    )?;

    fn test(
        condition: impl NdArray,
        x: impl NdArray,
        y: impl NdArray,
        expected: impl NdArray,
    ) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Where".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![
                    INPUT_X.to_string(),
                    INPUT_Y.to_string(),
                    INPUT_A.to_string(),
                ],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(condition, &Device::Cpu)?);
        inputs.insert(INPUT_Y.to_string(), Tensor::new(x, &Device::Cpu)?);
        inputs.insert(INPUT_A.to_string(), Tensor::new(y, &Device::Cpu)?);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval
            .get(OUTPUT_Z)
            .expect("Output 'z' not found")
            .to_dtype(DType::F64)?;

        let expected = Tensor::new(expected, &Device::Cpu)?.to_dtype(DType::F64)?;
        match expected.dims().len() {
            0 => assert_eq!(z.to_vec0::<f64>()?, expected.to_vec0::<f64>()?),
            1 => assert_eq!(z.to_vec1::<f64>()?, expected.to_vec1::<f64>()?),
            2 => assert_eq!(z.to_vec2::<f64>()?, expected.to_vec2::<f64>()?),
            3 => assert_eq!(z.to_vec3::<f64>()?, expected.to_vec3::<f64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

#[test]
fn test_floor() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Floor".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        // some values taken from https://numpy.org/doc/stable/reference/generated/numpy.floor.html
        vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.7,
            -1.5,
            -0.2,
            0.2,
            1.5,
            1.7,
            2.0,
        ],
        &[10],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec1::<f64>()?;

    assert!(results[0].is_nan());
    assert_eq!(
        results[1..],
        vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            -2.,
            -2.,
            -1.,
            0.,
            1.,
            1.,
            2.
        ]
    );

    Ok(())
}

#[test]
fn test_ceil() -> Result<()> {
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Ceil".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec![INPUT_X.to_string()],
            output: vec![OUTPUT_Z.to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        name: "".to_string(),
        initializer: vec![],
        input: vec![ValueInfoProto {
            name: INPUT_X.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        output: vec![ValueInfoProto {
            name: OUTPUT_Z.to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        value_info: vec![],
        doc_string: "".to_string(),
        sparse_initializer: vec![],
        quantization_annotation: vec![],
    }));
    let x = Tensor::from_vec(
        // some values taken from https://numpy.org/doc/stable/reference/generated/numpy.ceil.html
        vec![
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.7,
            -1.5,
            -0.2,
            0.2,
            1.5,
            1.7,
            2.0,
        ],
        &[10],
        &Device::Cpu,
    )?;

    let mut inputs: HashMap<String, Tensor> = HashMap::new();
    inputs.insert(INPUT_X.to_string(), x);

    let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
    assert_eq!(eval.len(), 1);

    let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

    let results = z.to_vec1::<f64>()?;

    assert!(results[0].is_nan());
    assert_eq!(
        results[1..],
        vec![
            f64::INFINITY,
            f64::NEG_INFINITY,
            -1.,
            -1.,
            -0.,
            1.,
            2.,
            2.,
            2.
        ]
    );

    Ok(())
}

// "ArgMin"
#[test]
fn test_argmin() -> Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-7
    // default_axes_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(1),
        None,
        &[[0i64, 0i64]],
    )?;
    // keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(1),
        Some(1),
        None,
        &[[1i64], [0i64]],
    )?;
    // // negative_axis_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(-1),
        Some(1),
        None,
        &[[1i64], [0i64]],
    )?;
    // no_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(0),
        None,
        &[0i64, 0i64],
    )?;
    // tests from https://pytorch.org/docs/stable/generated/torch.argmin.html#torch.argmin
    test(
        &[
            [0.1139, 0.2254, -0.1381, 0.3687],
            [1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240, 0.1207, -0.7506, -1.0213],
            [1.7809, -1.2960, 0.9384, 0.1438],
        ],
        Some(1),
        Some(0),
        None,
        &[2i64, 1i64, 3i64, 1i64],
    )?;
    test(
        &[
            [0.1139, 0.2254, -0.1381, 0.3687],
            [1.0100, -1.1975, -0.0102, -0.4732],
            [-0.9240, 0.1207, -0.7506, -1.0213],
            [1.7809, -1.2960, 0.9384, 0.1438],
        ],
        Some(1),
        None,
        None,
        &[[2i64], [1i64], [3i64], [1i64]],
    )?;
    fn test(
        data: impl NdArray,
        axis: Option<i64>,
        keepdims: Option<i64>,
        select_last_index: Option<i64>,
        expected: impl NdArray,
    ) -> Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis.unwrap_or(0),
            doc_string: "axis".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims.unwrap_or(1),
            doc_string: "keepdims".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_select_last_index = AttributeProto {
            name: "select_last_index".to_string(),
            ref_attr_name: "select_last_index".to_string(),
            i: select_last_index.unwrap_or(0),
            doc_string: "select_last_index".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![];
            if axis.is_some() {
                mut_attrs.push(att_axis);
            }
            if keepdims.is_some() {
                mut_attrs.push(att_keepdims);
            }
            if select_last_index.is_some() {
                mut_attrs.push(att_select_last_index);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ArgMin".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            1 => assert_eq!(z.to_vec1::<i64>()?, expected.to_vec1::<i64>()?),
            2 => assert_eq!(z.to_vec2::<i64>()?, expected.to_vec2::<i64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "ArgMax"
#[test]
fn test_argmax() -> Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-6
    // default_axes_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(1),
        None,
        &[[1i64, 1i64]],
    )?;
    // keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(1),
        Some(1),
        None,
        &[[0i64], [1i64]],
    )?;
    // // negative_axis_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        Some(-1),
        Some(1),
        None,
        &[[0i64], [1i64]],
    )?;
    // no_keepdims
    test(
        &[[2u32, 1u32], [3u32, 10u32]],
        None,
        Some(0),
        None,
        &[1i64, 1i64],
    )?;
    // tests from https://pytorch.org/docs/stable/generated/torch.argmax.html
    test(
        &[
            [1.3398, 0.2663, -0.2686, 0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092, 0.5419, -0.2993, 0.3195],
        ],
        Some(1),
        Some(0),
        None,
        &[0i64, 2i64, 0i64, 1i64],
    )?;
    test(
        &[
            [1.3398, 0.2663, -0.2686, 0.2450],
            [-0.7401, -0.8805, -0.3402, -1.1936],
            [0.4907, -1.3948, -1.0691, -0.3132],
            [-1.6092, 0.5419, -0.2993, 0.3195],
        ],
        Some(1),
        None,
        None,
        &[[0i64], [2i64], [0i64], [1i64]],
    )?;
    fn test(
        data: impl NdArray,
        axis: Option<i64>,
        keepdims: Option<i64>,
        select_last_index: Option<i64>,
        expected: impl NdArray,
    ) -> Result<()> {
        let att_axis = AttributeProto {
            name: "axis".to_string(),
            ref_attr_name: "axis".to_string(),
            i: axis.unwrap_or(0),
            doc_string: "axis".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_keepdims = AttributeProto {
            name: "keepdims".to_string(),
            ref_attr_name: "keepdims".to_string(),
            i: keepdims.unwrap_or(1),
            doc_string: "keepdims".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let att_select_last_index = AttributeProto {
            name: "select_last_index".to_string(),
            ref_attr_name: "select_last_index".to_string(),
            i: select_last_index.unwrap_or(0),
            doc_string: "select_last_index".to_string(),
            r#type: 2, // INT
            f: 0.0,
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![];
            if axis.is_some() {
                mut_attrs.push(att_axis);
            }
            if keepdims.is_some() {
                mut_attrs.push(att_keepdims);
            }
            if select_last_index.is_some() {
                mut_attrs.push(att_select_last_index);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "ArgMax".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        match expected.dims().len() {
            1 => assert_eq!(z.to_vec1::<i64>()?, expected.to_vec1::<i64>()?),
            2 => assert_eq!(z.to_vec2::<i64>()?, expected.to_vec2::<i64>()?),
            _ => unreachable!(),
        };

        Ok(())
    }

    Ok(())
}

// "LeakyRelu"
#[test]
fn test_leakyrelu() -> Result<()> {
    // tests from https://github.com/onnx/onnx/blob/main/docs/Operators.md#examples-80
    // leakyrelu
    test(&[-1.0, 0.0, 1.0], Some(0.1), &[-0.1, 0.0, 1.0])?;
    fn test(data: impl NdArray, alpha: Option<f32>, expected: impl NdArray) -> Result<()> {
        let att_alpha = AttributeProto {
            name: "alpha".to_string(),
            ref_attr_name: "alpha".to_string(),
            i: 0,
            doc_string: "alpha".to_string(),
            r#type: 1, // FLOAT
            f: alpha.unwrap_or(0.01),
            s: vec![],
            t: None,
            g: None,
            sparse_tensor: None,
            tp: None,
            floats: vec![],
            ints: vec![],
            strings: vec![],
            tensors: vec![],
            graphs: vec![],
            sparse_tensors: vec![],
            type_protos: vec![],
        };
        let attrs = {
            let mut mut_attrs = vec![];
            if alpha.is_some() {
                mut_attrs.push(att_alpha);
            }
            mut_attrs
        };
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "LeakyRelu".to_string(),
                domain: "".to_string(),
                attribute: attrs,
                input: vec![INPUT_X.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));
        let mut inputs: HashMap<String, Tensor> = HashMap::new();
        inputs.insert(INPUT_X.to_string(), Tensor::new(data, &Device::Cpu)?);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;
        for both in z
            .to_vec1::<f64>()?
            .iter()
            .zip(expected.to_vec1::<f64>()?.iter())
        {
            let (act, exp) = both;
            assert!(f64::abs(act - exp) < f32::EPSILON.into());
        }

        Ok(())
    }

    Ok(())
}

// "If"
#[test]
fn test_if() -> Result<()> {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    let output_type_proto = Some(TypeProto {
        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
            elem_type: DataType::Float.into(),
            shape: Some(TensorShapeProto {
                dim: vec![Dimension {
                    denotation: "".to_string(),
                    value: Some(dimension::Value::DimValue(5)),
                }],
            }),
        })),
        denotation: "".to_string(),
    });
    let then_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "then_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["then_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![x.len() as i64],
                    float_data: x.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let else_branch = GraphProto {
        output: vec![ValueInfoProto {
            name: "else_out".to_string(),
            r#type: output_type_proto.clone(),
            doc_string: "".to_string(),
        }],
        node: vec![NodeProto {
            op_type: "Constant".to_string(),
            input: vec![],
            output: vec!["else_out".to_string()],
            attribute: vec![AttributeProto {
                name: "value".to_string(),
                r#type: AttributeType::Tensor.into(),
                t: Some(TensorProto {
                    dims: vec![y.len() as i64],
                    float_data: y.clone(),
                    data_type: DataType::Float.into(),
                    ..TensorProto::default()
                }),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    };
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "If".to_string(),
            attribute: vec![
                AttributeProto {
                    name: "then_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(then_branch),
                    ..AttributeProto::default()
                },
                AttributeProto {
                    name: "else_branch".to_string(),
                    r#type: AttributeType::Graph.into(),
                    g: Some(else_branch),
                    ..AttributeProto::default()
                },
            ],
            input: vec!["cond".to_string()],
            output: vec!["res".to_string()],
            ..NodeProto::default()
        }],
        input: vec![],
        output: vec![ValueInfoProto {
            name: "res".to_string(),
            doc_string: "".to_string(),
            r#type: output_type_proto.clone(),
        }],
        ..GraphProto::default()
    }));

    for cond in [1u8, 0] {
        let inputs =
            HashMap::from_iter([("cond".to_string(), Tensor::full(cond, (1,), &Device::Cpu)?)]);
        let outputs = candle_onnx::simple_eval(&manual_graph, inputs)?;
        let expected = if cond != 0 { &x } else { &y };
        let Some(res) = outputs.get("res") else {
            candle::bail!("outputs didn't contain expected key `res`: {outputs:?}");
        };
        assert_eq!(&res.to_vec1::<f32>()?, expected);
    }
    Ok(())
}

#[test]
fn test_pad() -> Result<()> {
    let data = Tensor::from_vec(
        vec![
            1.0, 2.0, 3.0, //
            4.0, 5.0, 6.0, //
        ],
        (2, 3),
        &Device::Cpu,
    )?;
    let pads = Tensor::from_vec(vec![0i64, 1, 0, 0], (4,), &Device::Cpu)?;
    let mode = "reflect";

    let expected = Tensor::from_vec(
        vec![
            2.0, 1.0, 2.0, 3.0, //
            5.0, 4.0, 5.0, 6.0, //
        ],
        (2, 4),
        &Device::Cpu,
    )?;

    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        input: vec![
            ValueInfoProto {
                name: "data".to_string(),
                ..ValueInfoProto::default()
            },
            ValueInfoProto {
                name: "pads".to_string(),
                ..ValueInfoProto::default()
            },
        ],
        output: vec![ValueInfoProto {
            name: "output".to_string(),
            ..ValueInfoProto::default()
        }],
        node: vec![NodeProto {
            op_type: "Pad".to_string(),
            input: vec!["data".to_string(), "pads".to_string()],
            output: vec!["output".to_string()],
            attribute: vec![AttributeProto {
                name: "mode".to_string(),
                r#type: AttributeType::String.into(),
                s: mode.as_bytes().to_vec(),
                ..AttributeProto::default()
            }],
            ..NodeProto::default()
        }],
        ..GraphProto::default()
    }));

    let inputs = HashMap::from_iter([("data".to_string(), data), ("pads".to_string(), pads)]);
    let res = candle_onnx::simple_eval(&model, inputs)?;
    let Some(actual) = res.get("output") else {
        candle::bail!("outputs didn't contain expected key `output`: {res:?}");
    };

    assert_eq!(actual.to_vec2::<f64>()?, expected.to_vec2::<f64>()?);
    Ok(())
}

#[test]
fn test_slice() -> Result<()> {
    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Slice".to_string(),
            input: vec![
                "data".to_string(),
                "starts".to_string(),
                "ends".to_string(),
                "axes".to_string(),
                "steps".to_string(),
            ],
            output: vec!["result".to_string()],
            ..NodeProto::default()
        }],
        input: ["data", "starts", "ends", "axes", "steps"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        output: ["result"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        ..GraphProto::default()
    }));

    /*
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    axes = [0, 1]
    starts = [1, 0]
    ends = [2, 3]
    steps = [1, 2]
    result = [
        [5, 7],
    ]
    */

    let outputs = candle_onnx::simple_eval(
        &model,
        HashMap::from_iter([
            (
                "data".to_string(),
                Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8], (2, 4), &Device::Cpu)?,
            ),
            (
                "starts".to_string(),
                Tensor::from_vec(vec![1i64, 0], (2,), &Device::Cpu)?,
            ),
            (
                "ends".to_string(),
                Tensor::from_vec(vec![2i64, 3], (2,), &Device::Cpu)?,
            ),
            (
                "axes".to_string(),
                Tensor::from_vec(vec![0i64, 1], (2,), &Device::Cpu)?,
            ),
            (
                "steps".to_string(),
                Tensor::from_vec(vec![1i64, 2], (2,), &Device::Cpu)?,
            ),
        ]),
    )?;
    let actual = outputs.get("result").unwrap().to_vec2::<i64>()?;
    assert_eq!(actual, vec![vec![5i64, 7]]);

    /*
    data = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
    ]
    starts = [0, 1]
    ends = [-1, 1000]
    result = [
        [2, 3, 4],
    ]
    */
    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Slice".to_string(),
            input: vec!["data".to_string(), "starts".to_string(), "ends".to_string()],
            output: vec!["result".to_string()],
            ..NodeProto::default()
        }],
        input: ["data", "starts", "ends"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        output: ["result"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                r#type: None,
                doc_string: "".to_string(),
            })
            .collect(),
        ..GraphProto::default()
    }));
    let outputs = candle_onnx::simple_eval(
        &model,
        HashMap::from_iter([
            (
                "data".to_string(),
                Tensor::from_vec(vec![1i64, 2, 3, 4, 5, 6, 7, 8], (2, 4), &Device::Cpu)?,
            ),
            (
                "starts".to_string(),
                Tensor::from_vec(vec![0i64, 1], (2,), &Device::Cpu)?,
            ),
            (
                "ends".to_string(),
                Tensor::from_vec(vec![-1i64, 1000], (2,), &Device::Cpu)?,
            ),
        ]),
    )?;
    let actual = outputs.get("result").unwrap().to_vec2::<i64>()?;
    assert_eq!(actual, vec![vec![2i64, 3, 4]]);

    Ok(())
}

#[test]
fn test_lstm() -> Result<()> {
    // values generated from pytorch, so at least it's close enough to what pytorch does
    /*
    #!/usr/bin/env python3

    # torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)

    import torch

    rand_gen = torch.Generator()
    rand_gen.manual_seed(1)
    input_size = 3
    hidden_size = 5
    batch_size = 1
    sequence_length = 4
    number_directions = 1
    rnn = torch.nn.LSTM(input_size,hidden_size)
    weight_ih_l0 = torch.randn(rnn.weight_ih_l0.shape, generator=rand_gen)
    weight_hh_l0 = torch.randn(rnn.weight_hh_l0.shape, generator=rand_gen)
    bias_ih_l0 = torch.randn(rnn.bias_ih_l0.shape, generator=rand_gen)
    bias_hh_l0 = torch.randn(rnn.bias_hh_l0.shape, generator=rand_gen)
    rnn.weight_ih_l0 = torch.nn.Parameter(weight_ih_l0)
    rnn.weight_hh_l0 = torch.nn.Parameter(weight_hh_l0)
    rnn.bias_ih_l0 = torch.nn.Parameter(bias_ih_l0)
    rnn.bias_hh_l0 = torch.nn.Parameter(bias_hh_l0)
    input = torch.randn(sequence_length, batch_size, input_size, generator=rand_gen)
    h0 = torch.randn(number_directions, batch_size, hidden_size, generator=rand_gen)
    c0 = torch.randn(number_directions, batch_size, hidden_size, generator=rand_gen)
    output, (hn, cn) = rnn(input, (h0, c0))

    def fmt_tensor(t):
        return "Tensor::from_vec::<_, f32>(vec!"+  str(t.flatten().tolist()) + ", (" + "".join([str(n)+"," for n in t.shape])+"), &Device::Cpu)?"

    print("let input_size = ", input_size, ";")
    print("let hidden_size = ", hidden_size, ";")
    print("let batch_size = ", batch_size, ";")
    print("let sequence_length = ", sequence_length, ";")
    print("let number_directions = ", number_directions, ";")
    print("let weight_ih_l0 = ", fmt_tensor(rnn.weight_ih_l0), ";")
    print("let weight_hh_l0 = ", fmt_tensor(rnn.weight_hh_l0), ";")
    print("let bias_ih_l0 = ", fmt_tensor(rnn.bias_ih_l0), ";")
    print("let bias_hh_l0 = ", fmt_tensor(rnn.bias_hh_l0), ";")
    print("let input = ", fmt_tensor(input), ";")
    print("let h0 = ", fmt_tensor(h0), ";")
    print("let c0 = ", fmt_tensor(c0), ";")
    print("let output = ", fmt_tensor(output), ";")
    print("let hn = ", fmt_tensor(hn), ";")
    print("let cn = ", fmt_tensor(cn), ";")
    */
    let input_size = 3;
    let hidden_size = 5;
    let batch_size = 1;
    let sequence_length = 4;
    let number_directions = 1;
    let weight_ih_l0 = Tensor::from_vec::<_, f32>(
        vec![
            -1.525_595_9,
            -0.750_231_8,
            -0.653_980_9,
            -1.609_484_8,
            -0.100_167_18,
            -0.609_188_9,
            -0.979_772_27,
            -1.609_096_3,
            -0.712_144_6,
            0.303_722,
            -0.777_314_3,
            -0.251_455_25,
            -0.222_270_49,
            1.687_113_4,
            0.228_425_17,
            0.467_635_5,
            -0.696_972_4,
            -1.160_761_5,
            0.699_542_4,
            0.199_081_63,
            0.865_692_4,
            0.244_403_9,
            -0.662_911_36,
            0.807_308_26,
            1.101_680_6,
            -0.175_936_04,
            -2.245_557_8,
            -1.446_458,
            0.061_155_282,
            -0.617_744_45,
            -0.798_069_83,
            -0.131_623_21,
            1.879_345_8,
            -0.072_131_78,
            0.157_770_6,
            -0.773_454_9,
            0.199_056_5,
            0.045_702_778,
            0.152_956_92,
            -0.475_678_8,
            -0.111_019_83,
            0.292_735_25,
            -0.157_845_15,
            -0.028_787_14,
            0.453_254_58,
            1.142_161_1,
            0.248_610_7,
            -1.775_400_8,
            -0.025_502_462,
            -1.023_330_6,
            -0.596_185_15,
            -1.005_530_7,
            0.428_542_3,
            1.476_077_8,
            -1.786_867_9,
            1.610_317_6,
            -0.703_956_66,
            -0.185_265_8,
            -0.996_235_1,
            -0.831_255_26,
        ],
        (20, 3),
        &Device::Cpu,
    )?;
    let weight_hh_l0 = Tensor::from_vec::<_, f32>(
        vec![
            0.409_972_43,
            0.408_450_66,
            0.257_865_4,
            1.095_021_4,
            -0.506_486_6,
            0.099_775_404,
            -0.653_973_4,
            0.731_693_7,
            -1.456_733,
            1.608_935_4,
            0.093_769_975,
            -1.259_749,
            0.254_633_5,
            -0.501_957_3,
            -1.041_2,
            0.732_267_2,
            1.307_535_5,
            -1.162_798_8,
            0.119_636_11,
            -0.163_135_33,
            0.661_445_3,
            1.189_920_5,
            0.816_533_9,
            -0.913_523_6,
            -0.353_806_53,
            0.763_927_04,
            -0.588_950_7,
            -0.763_597_37,
            1.335_205_7,
            0.604_273_6,
            -0.103_442_08,
            -0.151_216_92,
            1.246_568_3,
            0.505_721_4,
            0.950_511_2,
            1.296_648_3,
            0.873_796_3,
            -0.560_259_4,
            1.285_784_5,
            0.816_823_84,
            -1.464_799_4,
            -1.262_928_4,
            1.122_018_8,
            1.566_334_1,
            2.558_138_4,
            -0.233_363_88,
            -0.013_472_13,
            1.860_634_8,
            1.549_620_5,
            0.347_629_25,
            0.093_008_03,
            0.614_740_3,
            0.712_364_55,
            -1.776_507_3,
            0.353_864_58,
            1.199_613_2,
            -0.712_258_93,
            -0.620_034_4,
            -0.228_134_95,
            -0.789_274_63,
            -1.611_111_8,
            -1.871_612_9,
            0.543_083_6,
            0.660_678_6,
            0.270_527_72,
            0.559_691_97,
            -0.318_396_3,
            1.511_720_7,
            -1.363_267_2,
            -0.983_219_6,
            1.511_266_7,
            0.641_870_74,
            -0.747_445_9,
            -0.923_438_55,
            0.573_398_4,
            -0.109_299_51,
            0.518_112_1,
            0.106_535_35,
            0.269_240_77,
            1.324_768,
            0.037_456_9,
            -0.637_839_3,
            -0.814_755_44,
            -0.689_506_53,
            0.843_654_3,
            1.165_701_3,
            0.526_932_2,
            1.619_253_3,
            -0.963_976_26,
            0.141_520_38,
            -0.163_660_96,
            -0.358_222_57,
            1.722_279_3,
            -0.303_575_6,
            0.238_874_2,
            1.344_001_2,
            0.103_225_69,
            1.100_354_2,
            -0.341_680_2,
            0.947_338_9,
        ],
        (20, 5),
        &Device::Cpu,
    )?;
    let bias_ih_l0 = Tensor::from_vec::<_, f32>(
        vec![
            -0.568_515_96,
            0.837_596_2,
            1.783_660_7,
            -0.195_424_66,
            0.235_193_13,
            1.914_243_3,
            1.836_411_1,
            1.324_532_4,
            -0.070_514_58,
            0.346_979_4,
            -0.653_679_6,
            1.558_620_2,
            0.218_566_15,
            -0.574_307_26,
            1.457_125_1,
            1.770_955_7,
            -2.017_3,
            0.423_503_2,
            0.573_022,
            -1.796_243,
        ],
        (20,),
        &Device::Cpu,
    )?;
    let bias_hh_l0 = Tensor::from_vec::<_, f32>(
        vec![
            1.247_040_4,
            1.273_851_2,
            0.390_949_25,
            0.387_210_5,
            0.144_403_95,
            0.777_168_45,
            -2.338_112_6,
            -0.829_120_4,
            1.166_139_1,
            1.478_657_5,
            0.267_608_73,
            0.756_119_85,
            -0.587_336_1,
            -2.061_920_6,
            0.430_473_48,
            0.337_656_62,
            -0.343_785_35,
            -0.617_226_06,
            1.252_969_3,
            -0.051_417_42,
        ],
        (20,),
        &Device::Cpu,
    )?;
    let input = Tensor::from_vec::<_, f32>(
        vec![
            0.647_212_8,
            -0.041_167_17,
            -0.177_493_08,
            -0.500_039_3,
            0.867_274_94,
            -0.273_192_23,
            -0.460_768_13,
            -0.099_093_71,
            0.472_844_8,
            1.004_948_5,
            -0.287_142_04,
            -1.161_862_1,
        ],
        (4, 1, 3),
        &Device::Cpu,
    )?;
    let h0 = Tensor::from_vec::<_, f32>(
        vec![
            0.027_581_785,
            0.565_238_24,
            -0.011_487_379,
            0.670_640_05,
            -0.492_925_05,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    let c0 = Tensor::from_vec::<_, f32>(
        vec![
            1.505_028_5,
            -2.326_355,
            1.616_89,
            -0.902_623_8,
            0.173_668_24,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    let output = Tensor::from_vec::<_, f32>(
        vec![
            0.595_601_7,
            -0.017_232_792,
            0.110_355_72,
            -0.493_231_74,
            0.047_632_16,
            0.635_845_2,
            0.040_328_12,
            -0.378_861_16,
            -0.746_434,
            0.200_809_09,
            0.584_026_5,
            0.145_328_82,
            -0.734_529_85,
            -0.521_430_43,
            0.219_038_17,
            0.742_045_16,
            0.319_438_8,
            -0.047_266_465,
            -0.282_384_96,
            0.271_313_4,
        ],
        (4, 1, 5),
        &Device::Cpu,
    )?;
    let hn = Tensor::from_vec::<_, f32>(
        vec![
            0.742_045_16,
            0.319_438_8,
            -0.047_266_465,
            -0.282_384_96,
            0.271_313_4,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    let cn = Tensor::from_vec::<_, f32>(
        vec![
            0.963_055_85,
            1.003_307,
            -1.754_899,
            -1.596_712_2,
            0.825_292_47,
        ],
        (1, 1, 5),
        &Device::Cpu,
    )?;
    // end of generated values

    let model = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "LSTM".to_string(),
            name: "LSTM_test".to_string(),
            attribute: vec![AttributeProto {
                name: "hidden_size".to_string(),
                r#type: AttributeType::Int.into(),
                i: hidden_size as i64,
                ..AttributeProto::default()
            }],
            input: vec![
                "input".to_string(),
                "w".to_string(),
                "r".to_string(),
                "b".to_string(), // b
                "".to_string(),  // seq_lens
                "h".to_string(),
                "c".to_string(),
            ],
            output: vec!["output".to_string(), "hn".to_string(), "cn".to_string()],
            ..NodeProto::default()
        }],
        input: ["input", "w", "r", "b", "h", "c"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        output: ["output", "hn", "cn"]
            .into_iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        ..GraphProto::default()
    }));
    // pytorch stores weight and bias as [ifco] but we want it as [iofc]
    // so we need to re-arrange the tensors a bit
    let idx_iofc = {
        let stride = hidden_size as i64;
        let dev = weight_ih_l0.device();
        let idx_i = Tensor::arange(0, stride, dev)?;
        let idx_f = Tensor::arange(stride, 2 * stride, dev)?;
        let idx_g = Tensor::arange(2 * stride, 3 * stride, dev)?;
        let idx_o = Tensor::arange(3 * stride, 4 * stride, dev)?;

        Tensor::cat(&[&idx_i, &idx_o, &idx_f, &idx_g], 0)?
    };
    let w = weight_ih_l0.index_select(&idx_iofc, 0)?;
    let w = w.reshape((number_directions, 4 * hidden_size, input_size))?;
    let r = weight_hh_l0.index_select(&idx_iofc, 0)?;
    let r = r.reshape((number_directions, 4 * hidden_size, hidden_size))?;
    let wb = bias_ih_l0.index_select(&idx_iofc, 0)?;
    let rb = bias_hh_l0.index_select(&idx_iofc, 0)?;
    let b = Tensor::cat(&[wb, rb], 0)?.reshape((number_directions, 8 * hidden_size))?;
    let output = output.reshape((sequence_length, number_directions, batch_size, hidden_size))?;
    let result = simple_eval(
        &model,
        HashMap::from_iter([
            ("input".to_string(), input),
            ("w".to_string(), w),
            ("r".to_string(), r),
            ("b".to_string(), b),
            ("h".to_string(), h0),
            ("c".to_string(), c0),
        ]),
    )?;
    let actual_output = result.get("output").unwrap();
    assert_eq!(output.dims(), actual_output.dims());
    let actual_hn = result.get("hn").unwrap();
    assert_eq!(hn.dims(), actual_hn.dims());
    let actual_cn = result.get("cn").unwrap();
    assert_eq!(cn.dims(), actual_cn.dims());
    let diff_close_enough = |a: &Tensor, b| -> Result<_> {
        let diffs = a.sub(b)?.flatten_all()?.to_vec1::<f32>()?;
        Ok(diffs.iter().all(|f| f.abs() < 0.0001))
    };
    assert!(
        diff_close_enough(&output, actual_output)?,
        "output did not match expected\n{actual_output}\n{output}",
    );
    assert!(
        diff_close_enough(&hn, actual_hn)?,
        "hn did not match expected\n{actual_hn}\n{hn}",
    );
    assert!(
        diff_close_enough(&cn, actual_cn)?,
        "cn did not match expected\n{actual_cn}\n{cn}",
    );

    Ok(())
}

#[test]
fn test_expand_dim_changed() -> Result<()> {
    // Create a manual graph for the Expand operation
    let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: "Expand".to_string(),
            domain: "".to_string(),
            attribute: vec![],
            input: vec!["data".to_string(), "new_shape".to_string()],
            output: vec!["expanded".to_string()],
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        input: vec![
            ValueInfoProto {
                name: "data".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
            ValueInfoProto {
                name: "new_shape".to_string(),
                doc_string: "".to_string(),
                r#type: None,
            },
        ],
        output: vec![ValueInfoProto {
            name: "expanded".to_string(),
            doc_string: "".to_string(),
            r#type: None,
        }],
        ..GraphProto::default()
    }));

    // Input tensor with shape [3, 1]
    let data = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], (3, 1), &Device::Cpu)?;

    // New shape tensor: [2, 1, 6]
    let new_shape = Tensor::from_vec(vec![2i64, 1, 6], (3,), &Device::Cpu)?;

    // Expected output after expansion
    let expected = Tensor::from_vec(
        vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32,
            2.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 3.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32,
            1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 3.0f32, 3.0f32, 3.0f32,
            3.0f32, 3.0f32, 3.0f32,
        ],
        (2, 3, 6),
        &Device::Cpu,
    )?;

    // Execute the model evaluation
    let inputs = HashMap::from_iter([
        ("data".to_string(), data),
        ("new_shape".to_string(), new_shape),
    ]);
    let result = candle_onnx::simple_eval(&manual_graph, inputs)?;

    // Retrieve and compare the result
    let expanded = result.get("expanded").expect("Output 'expanded' not found");

    assert_eq!(expanded.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);

    Ok(())
}

fn make_graph_helper(
    op_name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attribs: Vec<AttributeProto>,
) -> ModelProto {
    utils::create_model_proto_with_graph(Some(GraphProto {
        node: vec![NodeProto {
            op_type: op_name.to_string(),
            domain: "".to_string(),
            attribute: attribs,
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            name: "".to_string(),
            doc_string: "".to_string(),
        }],
        input: inputs
            .iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        output: outputs
            .iter()
            .map(|name| ValueInfoProto {
                name: name.to_string(),
                ..ValueInfoProto::default()
            })
            .collect(),
        ..GraphProto::default()
    }))
}

#[test]
fn test_expand_dim_unchanged() -> Result<()> {
    // Create a manual graph for the Expand operation
    let manual_graph = make_graph_helper("Expand", &["data", "new_shape"], &["expanded"], vec![]);

    // Input tensor with shape [3, 1] and dtype f32
    let data = Tensor::from_vec(vec![1.0f32, 2.0f32, 3.0f32], (3, 1), &Device::Cpu)?;

    // New shape tensor: [3, 4]
    let new_shape = Tensor::from_vec(vec![3i64, 4], (2,), &Device::Cpu)?;

    // Expected output after expansion, dtype f32
    let expected = Tensor::from_vec(
        vec![
            1.0f32, 1.0f32, 1.0f32, 1.0f32, 2.0f32, 2.0f32, 2.0f32, 2.0f32, 3.0f32, 3.0f32, 3.0f32,
            3.0f32,
        ],
        (3, 4),
        &Device::Cpu,
    )?;

    // Execute the model evaluation
    let inputs = HashMap::from_iter([
        ("data".to_string(), data),
        ("new_shape".to_string(), new_shape),
    ]);
    let result = candle_onnx::simple_eval(&manual_graph, inputs)?;

    // Retrieve and compare the result
    let expanded = result.get("expanded").expect("Output 'expanded' not found");
    assert_eq!(expanded.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);

    Ok(())
}

fn make_split_graph_helper(inputs: &[&str], outputs: &[&str], axis: i64) -> ModelProto {
    let attribs = vec![AttributeProto {
        name: "axis".to_string(),
        r#type: AttributeType::Int.into(),
        i: axis,
        ..AttributeProto::default()
    }];

    make_graph_helper("Split", inputs, outputs, attribs)
}

#[test]
fn test_split_equal_parts_1d_opset13() -> Result<()> {
    let input = Tensor::from_vec(
        vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32],
        (6,),
        &Device::Cpu,
    )?;
    let mut inputs = HashMap::new();
    inputs.insert("input".to_string(), input);

    {
        let manual_graph =
            make_split_graph_helper(&["input"], &["output_1", "output_2", "output_3"], 0);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs.clone())?;
        assert_eq!(eval.len(), 3);

        let out1 = eval.get("output_1").expect("Output 'output_1' not found");
        let out2 = eval.get("output_2").expect("Output 'output_2' not found");
        let out3 = eval.get("output_3").expect("Output 'output_3' not found");

        assert_eq!(out1.to_vec1::<f32>()?, vec![1.0f32, 2.0f32]);
        assert_eq!(out2.to_vec1::<f32>()?, vec![3.0f32, 4.0f32]);
        assert_eq!(out3.to_vec1::<f32>()?, vec![5.0f32, 6.0f32]);
    }

    {
        let splits = Tensor::from_vec(vec![2i64, 4], (2,), &Device::Cpu)?;
        inputs.insert("split".to_string(), splits);

        let manual_graph =
            make_split_graph_helper(&["input", "split"], &["output_1", "output_2"], 0);
        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 2);

        let out1 = eval.get("output_1").expect("Output 'output_1' not found");
        let out2 = eval.get("output_2").expect("Output 'output_2' not found");

        assert_eq!(out1.to_vec1::<f32>()?, vec![1.0f32, 2.0f32]);
        assert_eq!(out2.to_vec1::<f32>()?, vec![3.0f32, 4.0f32, 5.0f32, 6.0f32]);
    }
    Ok(())
}

fn make_reduce_sum_graph_helper(
    inputs: &[&str],
    outputs: &[&str],
    keepdims: Option<i64>,
    noop_with_empty_axes: Option<i64>,
) -> ModelProto {
    let mut attribs = vec![];
    if let Some(keepdims) = keepdims {
        attribs.push(AttributeProto {
            name: "keepdims".to_string(),
            r#type: AttributeType::Int.into(),
            i: keepdims,
            ..AttributeProto::default()
        });
    }
    if let Some(noop_with_empty_axes) = noop_with_empty_axes {
        attribs.push(AttributeProto {
            name: "noop_with_empty_axes".to_string(),
            r#type: AttributeType::Ints.into(),
            i: noop_with_empty_axes,
            ..AttributeProto::default()
        });
    }
    make_graph_helper("ReduceSum", inputs, outputs, attribs)
}

#[test]
fn test_reduce_sum_default_axes_keepdims() -> Result<()> {
    let manual_graph = make_reduce_sum_graph_helper(&["data", "axes"], &["reduced"], Some(1), None);

    // Test with example data
    {
        let data = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;
        // let axes = Tensor::from_vec(Vec::<i64>::new(), (0,), &Device::Cpu)?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        // inputs.insert("axes".to_string(), axes);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");
        let expected = Tensor::from_vec(vec![78.0f32], (1, 1, 1), &Device::Cpu)?;

        assert_eq!(reduced.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);
    }

    {
        let data = Tensor::from_vec(
            vec![
                -5.2f32, 7.8, -3.1, 9.4, 2.6, -8.7, 4.3, -1.9, 6.5, -0.8, -7.2, 3.6,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data.clone());

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");
        let expected = data.sum_all()?.reshape((1, 1, 1))?;

        assert_eq!(reduced.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);
    }

    Ok(())
}

#[test]
fn test_reduce_sum_do_not_keep_dims() -> Result<()> {
    let manual_graph = make_reduce_sum_graph_helper(&["data", "axes"], &["reduced"], Some(0), None);

    // Test with example data
    {
        let data = Tensor::from_vec(
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;
        let axes = Tensor::from_vec(vec![1i64], (1,), &Device::Cpu)?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data);
        inputs.insert("axes".to_string(), axes);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");
        let expected = Tensor::from_vec(
            vec![4.0f32, 6.0, 12.0, 14.0, 20.0, 22.0],
            (3, 2),
            &Device::Cpu,
        )?;

        assert_eq!(reduced.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    }

    // Test with random data
    {
        let _shape = (3, 2, 2);
        let data = Tensor::from_vec(
            vec![
                -5.2f32, 7.8, -3.1, 9.4, 2.6, -8.7, 4.3, -1.9, 6.5, -0.8, -7.2, 3.6,
            ],
            (3, 2, 2),
            &Device::Cpu,
        )?;
        let axes = Tensor::from_vec(vec![1i64], (1,), &Device::Cpu)?;

        let mut inputs = HashMap::new();
        inputs.insert("data".to_string(), data.clone());
        inputs.insert("axes".to_string(), axes);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let reduced = eval.get("reduced").expect("Output 'reduced' not found");

        // Calculate expected result
        let expected = data.sum(1)?;

        assert_eq!(reduced.to_vec2::<f32>()?, expected.to_vec2::<f32>()?);
    }

    Ok(())
}

// Xor
#[test]
fn test_xor() -> Result<()> {
    // tests based on: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor xor

    // 2d
    test(
        &[[0_u8, 1, 0, 0], [0, 0, 1, 1], [0, 1, 1, 1]],
        &[[1_u8, 1, 0, 0], [1, 0, 0, 1], [1, 1, 1, 0]],
        &[[1_u8, 0, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1]],
    )?;

    // 3d
    test(
        &[
            [
                [0_u8, 1, 1, 1, 1],
                [0, 1, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 1],
            ],
            [
                [0, 0, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            [
                [1, 0, 0, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 0, 1],
                [1, 0, 0, 0, 1],
            ],
        ],
        &[
            [
                [1_u8, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
                [1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [1, 0, 0, 1, 1],
                [1, 0, 1, 1, 1],
                [0, 1, 0, 1, 1],
                [1, 1, 1, 0, 0],
            ],
            [
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 0, 1, 0],
            ],
        ],
        &[
            [
                [1_u8, 1, 1, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 1, 0, 1],
                [0, 0, 0, 0, 1],
            ],
            [
                [1, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 1, 1, 0],
            ],
            [
                [1, 1, 1, 0, 1],
                [0, 0, 1, 1, 0],
                [1, 0, 1, 1, 1],
                [0, 1, 0, 1, 1],
            ],
        ],
    )?;

    // 4d
    test(
        &[
            [
                [[0_u8, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 1]],
                [[1, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
            ],
            [
                [[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 0]],
                [[1, 0, 0, 1], [1, 0, 1, 1], [1, 1, 0, 1]],
            ],
        ],
        &[
            [
                [[1_u8, 0, 1, 0], [0, 0, 1, 1], [1, 0, 1, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]],
            ],
            [
                [[1, 1, 1, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
                [[0, 0, 0, 0], [1, 0, 0, 0], [1, 1, 1, 1]],
            ],
        ],
        &[
            [
                [[1_u8, 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 1]],
                [[1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]],
            ],
            [
                [[0, 0, 1, 0], [1, 0, 1, 1], [1, 0, 1, 0]],
                [[1, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0]],
            ],
        ],
    )?;

    // tests based on: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Xor xor_broadcast
    // 3d vs 1d
    test(
        // Shape (3, 4, 5)
        &[
            [
                [0_u8, 0, 0, 0, 1],
                [0, 1, 0, 1, 1],
                [1, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
            ],
            [
                [0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [
                [1, 1, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [1, 1, 0, 1, 1],
            ],
        ],
        // shape (5)
        &[1_u8, 0, 0, 1, 1],
        // shape (3, 4, 5)
        &[
            [
                [1_u8, 0, 0, 1, 0],
                [1, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0],
            ],
            [
                [1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0],
                [1, 1, 1, 0, 1],
                [1, 0, 0, 1, 0],
            ],
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [1, 1, 1, 1, 0],
                [0, 1, 0, 0, 0],
            ],
        ],
    )?;

    // 3d vs 2d
    test(
        // Shape (3, 4, 5)
        &[
            [
                [0_u8, 0, 0, 0, 1],
                [0, 1, 0, 1, 1],
                [1, 0, 0, 1, 1],
                [0, 0, 1, 0, 1],
            ],
            [
                [0, 1, 0, 1, 1],
                [1, 1, 0, 0, 1],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
            [
                [1, 1, 0, 1, 1],
                [0, 0, 0, 1, 1],
                [0, 1, 1, 0, 1],
                [1, 1, 0, 1, 1],
            ],
        ],
        // shape (4, 5)
        &[
            [0_u8, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 0],
        ],
        // shape (3, 4, 5)
        &[
            [
                [0_u8, 1, 0, 1, 1],
                [0, 1, 1, 1, 1],
                [0, 1, 0, 0, 0],
                [1, 1, 1, 1, 1],
            ],
            [
                [0, 0, 0, 0, 1],
                [1, 1, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 1, 0, 1, 1],
            ],
            [
                [1, 0, 0, 0, 1],
                [0, 0, 1, 1, 1],
                [1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
            ],
        ],
    )?;

    // 4d vs 2d
    test(
        // Shape (2, 3, 3, 4)
        &[
            [
                [[1_u8, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
                [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
                [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
            ],
            [
                [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]],
                [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]],
                [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
            ],
        ],
        // shape (3, 4)
        &[[0_u8, 0, 1, 1], [1, 1, 1, 1], [0, 1, 0, 1]],
        // shape (2, 3, 3, 4)
        &[
            [
                [[1_u8, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
                [[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 0]],
                [[1, 0, 1, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
            ],
            [
                [[0, 1, 1, 0], [0, 0, 1, 0], [1, 1, 1, 0]],
                [[1, 1, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]],
                [[1, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]],
            ],
        ],
    )?;

    // 4d vs 3d
    test(
        // Shape (2, 3, 3, 4)
        &[
            [
                [[1_u8, 0, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0]],
                [[1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]],
                [[1, 0, 0, 0], [1, 1, 1, 0], [0, 0, 1, 1]],
            ],
            [
                [[0, 1, 0, 1], [1, 1, 0, 1], [1, 0, 1, 1]],
                [[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1]],
                [[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]],
            ],
        ],
        // shape (3, 3, 4)
        &[
            [[1_u8, 1, 0, 0], [0, 0, 1, 1], [0, 1, 0, 0]],
            [[0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]],
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1]],
        ],
        // shape (2, 3, 3, 4)
        &[
            [
                [[0_u8, 1, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]],
                [[1, 0, 0, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
                [[1, 1, 1, 0], [0, 1, 0, 1], [1, 1, 1, 0]],
            ],
            [
                [[1, 0, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
                [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 1, 0]],
                [[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0]],
            ],
        ],
    )?;

    // 4d vs 4d
    test(
        // Shape (1, 4, 1, 2)
        &[[[[1_u8, 0]], [[1, 0]], [[1, 0]], [[1, 1]]]],
        // shape (2, 1, 4, 2)
        &[
            [[[0_u8, 0], [1, 1], [1, 1], [1, 1]]],
            [[[0, 1], [1, 0], [0, 1], [0, 0]]],
        ],
        // shape (2, 4, 4, 2)
        &[
            [
                [[1_u8, 0], [0, 1], [0, 1], [0, 1]],
                [[1, 0], [0, 1], [0, 1], [0, 1]],
                [[1, 0], [0, 1], [0, 1], [0, 1]],
                [[1, 1], [0, 0], [0, 0], [0, 0]],
            ],
            [
                [[1, 1], [0, 0], [1, 1], [1, 0]],
                [[1, 1], [0, 0], [1, 1], [1, 0]],
                [[1, 1], [0, 0], [1, 1], [1, 0]],
                [[1, 0], [0, 1], [1, 0], [1, 1]],
            ],
        ],
    )?;

    fn test(input: impl NdArray, other: impl NdArray, expected: impl NdArray) -> Result<()> {
        let manual_graph = utils::create_model_proto_with_graph(Some(GraphProto {
            node: vec![NodeProto {
                op_type: "Xor".to_string(),
                domain: "".to_string(),
                attribute: vec![],
                input: vec![INPUT_X.to_string(), INPUT_Y.to_string()],
                output: vec![OUTPUT_Z.to_string()],
                name: "".to_string(),
                doc_string: "".to_string(),
            }],
            name: "".to_string(),
            initializer: vec![],
            input: vec![],
            output: vec![ValueInfoProto {
                name: OUTPUT_Z.to_string(),
                doc_string: "".to_string(),
                r#type: None,
            }],
            value_info: vec![],
            doc_string: "".to_string(),
            sparse_initializer: vec![],
            quantization_annotation: vec![],
        }));

        let inputs: HashMap<String, Tensor> = HashMap::from([
            (INPUT_X.to_string(), Tensor::new(input, &Device::Cpu)?),
            (INPUT_Y.to_string(), Tensor::new(other, &Device::Cpu)?),
        ]);

        let eval = candle_onnx::simple_eval(&manual_graph, inputs)?;
        assert_eq!(eval.len(), 1);

        let z = eval.get(OUTPUT_Z).expect("Output 'z' not found");

        let expected = Tensor::new(expected, &Device::Cpu)?;

        match expected.dims().len() {
            0 => {
                assert_eq!(z.to_vec0::<u8>()?, expected.to_vec0::<u8>()?)
            }
            1 => {
                assert_eq!(z.to_vec1::<u8>()?, expected.to_vec1::<u8>()?)
            }
            2 => {
                assert_eq!(z.to_vec2::<u8>()?, expected.to_vec2::<u8>()?)
            }
            3 => {
                assert_eq!(z.to_vec3::<u8>()?, expected.to_vec3::<u8>()?)
            }
            4 => {
                // Candle has no method equivallent to `to_vec4()`
                // So, as a hack, we flatten it to a single dim vec to test the results
                assert_eq!(
                    z.flatten_all()?.to_vec1::<u8>()?,
                    expected.flatten_all()?.to_vec1::<u8>()?
                )
            }
            _ => unreachable!(),
        };

        Ok(())
    }
    Ok(())
}
