//! Recurrent Neural Networks
use candle::{DType, Device, Result, Tensor};

/// Trait for Recurrent Neural Networks.
#[allow(clippy::upper_case_acronyms)]
pub trait RNN {
    type State;

    /// A zero state from which the recurrent network is usually initialized.
    fn zero_state(&self, batch_dim: usize) -> Result<Self::State>;

    /// Applies a single step of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, features].
    fn step(&self, input: &Tensor, state: &Self::State) -> Result<Self::State>;

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    /// The initial state is the result of applying zero_state.
    fn seq(&self, input: &Tensor) -> Result<(Tensor, Self::State)> {
        let batch_dim = input.dim(0)?;
        let state = self.zero_state(batch_dim)?;
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, state: &Self::State) -> Result<(Tensor, Self::State)>;
}

/// The state for a LSTM network, this contains two tensors.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
pub struct LSTMState {
    h: Tensor,
    c: Tensor,
}

impl LSTMState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    /// The cell state vector.
    pub fn c(&self) -> &Tensor {
        &self.c
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct LSTMConfig {
    pub has_biases: bool,
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            has_biases: true,
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
        }
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
#[allow(clippy::upper_case_acronyms, unused)]
#[derive(Debug)]
pub struct LSTM {
    flat_weights: Vec<Tensor>,
    hidden_dim: usize,
    config: LSTMConfig,
    device: Device,
    dtype: DType,
}

/// Creates a LSTM layer.
pub fn lstm(
    _in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: crate::VarBuilder,
) -> Result<LSTM> {
    let flat_weights = vec![]; // TODO
    Ok(LSTM {
        flat_weights,
        hidden_dim,
        config,
        device: vb.device().clone(),
        dtype: vb.dtype(),
    })
}

impl RNN for LSTM {
    type State = LSTMState;

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let zeros = Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?;
        Ok(Self::State {
            h: zeros.clone(),
            c: zeros.clone(),
        })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let input = input.unsqueeze(1)?;
        let (_output, state) = self.seq_init(&input, in_state)?;
        Ok(state)
    }

    fn seq_init(&self, _input: &Tensor, _in_state: &Self::State) -> Result<(Tensor, Self::State)> {
        todo!()
    }
}
