use crate::onnx::NodeProto;
use candle::Tensor;
use std::collections::HashMap;

//This struct is used to represent a node in the computation graph
//The idea is not to use the NodeProto directly in the computation graph
//On a longer term, this can lead to a more optimized representation of the computation graph.
//For now, it is just a wrapper around the NodeProto and the context
pub struct ComputeNode<'a> {
    node_proto: &'a NodeProto,
    context: &'a HashMap<String, Tensor>,
}

impl<'a> ComputeNode<'a> {
    pub fn new(node_proto: &'a NodeProto, context: &'a HashMap<String, Tensor>) -> Self {
        ComputeNode {
            node_proto,
            context,
        }
    }

    pub fn get_input(&self, index: usize) -> Option<&Tensor> {
        let input_name = self.node_proto.input.get(index)?;
        self.context.get(input_name)
    }

    pub fn get_output(&self, index: usize) -> Option<&String> {
        self.node_proto.output.get(index)
    }
}
