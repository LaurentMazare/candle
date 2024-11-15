//! mimi model
//!
//! mimi is a neural net model for speech generation and conversion.
//!
//! - [GH Link](https://github.com/kyutai-labs/mimi)
//!

// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
pub use candle;
pub use candle_nn;

pub mod conv;
pub mod encodec;
pub mod quantization;
pub mod seanet;
pub mod transformer;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

pub use encodec::{load, Config, Encodec as Model};
