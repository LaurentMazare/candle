pub mod onnxop;
pub use onnxop::{OnnxOp, OnnxOpError, OnnxOpRegistry, OpOutput};

pub mod compute_node;
pub use compute_node::ComputeNode;

mod math;
use math::sign;

pub fn registry() -> Result<OnnxOpRegistry, OnnxOpError> {
    let mut registry = OnnxOpRegistry::new();
    registry.insert("Sign", Box::new(sign::Sign))?;
    Ok(registry)
}
