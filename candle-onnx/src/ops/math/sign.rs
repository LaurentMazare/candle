use crate::ops::compute_node::ComputeNode;
use crate::ops::OnnxOpError::ComputationFailed;
use crate::ops::{OnnxOp, OnnxOpError, OpOutput};

pub(crate) struct Sign;
impl OnnxOp for Sign {
    fn eval(&self, node: &ComputeNode) -> Result<OpOutput, OnnxOpError> {
        let input = node
            .get_input(0)
            .ok_or_else(|| ComputationFailed("input 0 not found".to_string()))?;

        let output = input
            .sign()
            .map_err(|err| ComputationFailed(format!("{:?}", err)))?;

        let output_name = node
            .get_output(0)
            .ok_or_else(|| ComputationFailed("output 0 not found".to_string()))?;

        Ok((output_name.clone(), output))
    }
}
