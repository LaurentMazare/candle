use crate::onnx;
use crate::onnx::attribute_proto::AttributeType;
use crate::onnx::tensor_proto::DataType;
use candle::{bail, DType, Device, Result, Tensor};
use std::collections::HashMap;

pub type Value = Tensor;

pub fn dtype(dt: DataType) -> Option<DType> {
    match dt {
        DataType::Uint8 => Some(DType::U8),
        DataType::Uint32 => Some(DType::U32),
        DataType::Int64 => Some(DType::I64),
        DataType::Float16 => Some(DType::F16),
        DataType::Float => Some(DType::F32),
        DataType::Double => Some(DType::F64),
        _ => None,
    }
}

trait Attr {
    const TYPE: AttributeType;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self>;
}

impl Attr for i64 {
    const TYPE: AttributeType = AttributeType::Int;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(&attr.i)
    }
}

impl Attr for [i64] {
    const TYPE: AttributeType = AttributeType::Ints;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        Ok(attr.ints.as_slice())
    }
}

impl Attr for str {
    const TYPE: AttributeType = AttributeType::String;
    fn get(attr: &onnx::AttributeProto) -> Result<&Self> {
        std::str::from_utf8(&attr.s).map_err(candle::Error::wrap)
    }
}

fn get_attr_<'a>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a onnx::AttributeProto> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => {
            bail!(
                "cannot find the '{name}' attribute in '{}' for {}",
                node.op_type,
                node.name
            )
        }
        Some(dt) => Ok(dt),
    }
}

fn get_attr<'a, T: Attr + ?Sized>(node: &'a onnx::NodeProto, name: &str) -> Result<&'a T> {
    let attr = get_attr_(node, name)?;
    if attr.r#type() != T::TYPE {
        bail!(
            "unsupported type {:?} for '{name}' attribute in '{}' for {}",
            attr.r#type,
            node.op_type,
            node.name
        )
    }
    T::get(attr)
}

fn get_attr_opt<'a, T: Attr + ?Sized>(
    node: &'a onnx::NodeProto,
    name: &str,
) -> Result<Option<&'a T>> {
    match node.attribute.iter().find(|attr| attr.name == name) {
        None => Ok(None),
        Some(attr) => {
            if attr.r#type() != T::TYPE {
                bail!(
                    "unsupported type {:?} for '{name}' attribute in '{}' for {}",
                    attr.r#type,
                    node.op_type,
                    node.name
                )
            }
            let val = T::get(attr)?;
            Ok(Some(val))
        }
    }
}

// This function provides a direct evaluation of the proto.
// Longer-term, we should first convert the proto to an intermediate representation of the compute
// graph so as to make multiple evaluations more efficient.
// An example upside of this would be to remove intermediary values when they are not needed
// anymore.
pub fn simple_eval(
    model: &onnx::ModelProto,
    inputs: HashMap<String, Value>,
) -> Result<HashMap<String, Value>> {
    let graph = match &model.graph {
        None => bail!("no graph defined in proto"),
        Some(graph) => graph,
    };
    // TODO: validate the inputs.
    let mut values = inputs;
    // The nodes are topologically sorted so we can just process them in order.
    for node in graph.node.iter() {
        let get = |input_name: &str| match values.get(input_name) {
            Some(value) => Ok(value),
            None => bail!("cannot find {input_name} for op {}", node.name),
        };
        // TODO: Validate node.input for each operator.
        match node.op_type.as_str() {
            "Add" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_add(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Sub" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_sub(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Mul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_mul(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Div" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_div(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Equal" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.eq(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "MatMul" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?;
                let output = input0.broadcast_matmul(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "Reshape" => {
                let input0 = get(&node.input[0])?;
                let input1 = get(&node.input[1])?.to_vec1::<i64>()?;
                // TODO: Check that there is at most a single -1, handle other neg values.
                let input1 = input1
                    .iter()
                    .map(|&v| {
                        if v == -1 {
                            input0.elem_count()
                        } else {
                            v as usize
                        }
                    })
                    .collect::<Vec<usize>>();
                let output = input0.reshape(input1)?;
                values.insert(node.output[0].clone(), output);
            }
            "LogSoftmax" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => candle_nn::ops::softmax_last_dim(input)?,
                    Some(&axis) => {
                        let num_axis = input.rank() as i64;
                        let axis = if axis >= 0 {
                            axis as usize
                        } else if axis < -num_axis {
                            bail!("wrong axis in concat {axis} for shape {:?}", input.shape())
                        } else {
                            (num_axis - axis) as usize
                        };
                        candle_nn::ops::log_softmax(input, axis)?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Softmax" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<i64>(node, "axis")? {
                    None => candle_nn::ops::softmax_last_dim(input)?,
                    Some(&axis) => {
                        let num_axis = input.rank() as i64;
                        let axis = if axis >= 0 {
                            axis as usize
                        } else if axis < -num_axis {
                            bail!("wrong axis in concat {axis} for shape {:?}", input.shape())
                        } else {
                            (num_axis - axis) as usize
                        };
                        candle_nn::ops::softmax(input, axis)?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Transpose" => {
                let input = get(&node.input[0])?;
                let output = match get_attr_opt::<[i64]>(node, "perm")? {
                    None => input.t()?,
                    Some(perm) => {
                        let perm = perm.iter().map(|&v| v as usize).collect::<Vec<_>>();
                        input.permute(perm)?
                    }
                };
                values.insert(node.output[0].clone(), output);
            }
            "Conv" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv
                let dilations = get_attr_opt::<[i64]>(node, "dilations")?;
                let groups = get_attr_opt::<i64>(node, "group")?.copied().unwrap_or(1);
                let _kernel_shape = get_attr_opt::<[i64]>(node, "kernel_shape")?;
                let pads = get_attr_opt::<[i64]>(node, "pads")?;
                let strides = get_attr_opt::<[i64]>(node, "strides")?;
                let auto_pad = get_attr_opt::<str>(node, "auto_pad")?;
                match auto_pad {
                    None | Some("NOTSET") => (),
                    Some(s) => bail!("unsupported auto_pad {s}"),
                };
                let xs = get(&node.input[0])?;
                let ws = get(&node.input[1])?;
                let ys = match ws.rank() {
                    3 => {
                        let pads = match pads {
                            None => 0,
                            Some([p]) => *p as usize,
                            Some(pads) => {
                                bail!("more pads than expected in conv1d {pads:?} {}", node.name)
                            }
                        };
                        let strides = match strides {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                bail!("more strides than expected in conv1d {s:?} {}", node.name)
                            }
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some(s) => {
                                bail!("more dilations than expected in conv1d {s:?} {}", node.name)
                            }
                        };
                        xs.conv1d(ws, pads, strides, dilations, groups as usize)?
                    }
                    4 => {
                        let pads = match pads {
                            None => 0,
                            Some([p]) => *p as usize,
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    bail!("pads to be the same on both axis {pads:?} {}", node.name)
                                }
                                *p1 as usize
                            }
                            Some(pads) => {
                                bail!("more pads than expected in conv2d {pads:?} {}", node.name)
                            }
                        };
                        let strides = match strides {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    bail!(
                                        "strides to be the same on both axis {pads:?} {}",
                                        node.name
                                    )
                                }
                                *p1 as usize
                            }
                            Some(s) => {
                                bail!("more strides than expected in conv2d {s:?} {}", node.name)
                            }
                        };
                        let dilations = match dilations {
                            None => 1,
                            Some([p]) => *p as usize,
                            Some([p1, p2]) => {
                                if p1 != p2 {
                                    bail!(
                                        "dilations to be the same on both axis {pads:?} {}",
                                        node.name
                                    )
                                }
                                *p1 as usize
                            }
                            Some(s) => {
                                bail!("more dilations than expected in conv2d {s:?} {}", node.name)
                            }
                        };
                        xs.conv2d(ws, pads, strides, dilations, groups as usize)?
                    }
                    rank => bail!(
                        "unsupported rank for weight matrix {rank} in conv {}",
                        node.name
                    ),
                };
                let ys = if node.input.len() > 2 {
                    let bs = get(&node.input[2])?;
                    ys.broadcast_add(bs)?
                } else {
                    ys
                };
                values.insert(node.output[0].clone(), ys);
            }
            "Concat" => {
                // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Concat
                let inputs = node
                    .input
                    .iter()
                    .map(|n| Ok(get(n.as_str())?.clone()))
                    .collect::<Result<Vec<Value>>>()?;
                let axis: i64 = *get_attr(node, "axis")?;
                let num_axis = if inputs.is_empty() {
                    bail!("empty concat")
                } else {
                    inputs[0].rank() as i64
                };
                let axis = if axis >= 0 {
                    axis as usize
                } else if axis < -num_axis {
                    bail!(
                        "wrong axis in concat {axis} for shape {:?}",
                        inputs[0].shape()
                    )
                } else {
                    (num_axis - axis) as usize
                };
                let output = Tensor::cat(&inputs, axis)?;
                values.insert(node.output[0].clone(), output);
            }
            "Abs" => {
                let input = get(&node.input[0])?;
                let output = input.abs()?;
                values.insert(node.output[0].clone(), output);
            }
            "Cos" => {
                let input = get(&node.input[0])?;
                let output = input.cos()?;
                values.insert(node.output[0].clone(), output);
            }
            "Sin" => {
                let input = get(&node.input[0])?;
                let output = input.sin()?;
                values.insert(node.output[0].clone(), output);
            }
            "Neg" => {
                let input = get(&node.input[0])?;
                let output = input.neg()?;
                values.insert(node.output[0].clone(), output);
            }
            "Erf" => {
                let input = get(&node.input[0])?;
                let output = input.erf()?;
                values.insert(node.output[0].clone(), output);
            }
            "Tanh" => {
                let input = get(&node.input[0])?;
                let output = input.tanh()?;
                values.insert(node.output[0].clone(), output);
            }
            "Sigmoid" => {
                let input = get(&node.input[0])?;
                let output = candle_nn::ops::sigmoid(input)?;
                values.insert(node.output[0].clone(), output);
            }
            "Gelu" => {
                let input = get(&node.input[0])?;
                let output = input.gelu_erf()?;
                values.insert(node.output[0].clone(), output);
            }
            "Relu" => {
                let input = get(&node.input[0])?;
                let output = input.relu()?;
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Constant
            "Constant" => {
                let value = match node.attribute.iter().find(|attr| attr.name == "value") {
                    None => {
                        // TODO: support sparse_value etc.
                        bail!("cannot find 'value' attr in 'Constant' for {}", node.name)
                    }
                    Some(value) => value,
                };
                let output = match value.r#type() {
                    AttributeType::Tensor => {
                        let t = value.t.as_ref().unwrap();
                        let dims: Vec<usize> = t.dims.iter().map(|&x| x as usize).collect();
                        match DataType::try_from(t.data_type) {
                            Ok(dt) => match dtype(dt) {
                                Some(dt) => Tensor::from_raw_buffer(
                                    t.raw_data.as_slice(),
                                    dt,
                                    dims.as_slice(),
                                    &Device::Cpu,
                                )?,
                                None => {
                                    bail!("unsupported 'value' data-type {dt:?} for {}", node.name)
                                }
                            },
                            Err(_) => {
                                bail!(
                                    "unsupported 'value' data-type {} for {}",
                                    t.data_type,
                                    node.name
                                )
                            }
                        }
                    }
                    rtype => bail!("unsupported 'value' type {rtype:?} for {}", node.name),
                };
                values.insert(node.output[0].clone(), output);
            }
            // https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
            "Cast" => {
                let input = get(&node.input[0])?;
                let dt: i64 = *get_attr(node, "to")?;
                let dtype = match DataType::try_from(dt as i32) {
                    Ok(dt) => match dtype(dt) {
                        Some(dt) => dt,
                        None => {
                            bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                        }
                    },
                    Err(_) => {
                        bail!("unsupported 'to' value {dt:?} for cast {}", node.name)
                    }
                };
                let output = input.to_dtype(dtype)?;
                values.insert(node.output[0].clone(), output);
            }
            op_type => bail!("unsupported op_type {op_type} for op {}", node.name),
        }
    }
    graph
        .output
        .iter()
        .map(|output| match values.remove(&output.name) {
            None => bail!("cannot find output {}", output.name),
            Some(value) => Ok((output.name.clone(), value)),
        })
        .collect()
}
