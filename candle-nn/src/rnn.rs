//! Recurrent Neural Networks
use candle::{DType, Device, IndexOp, Result, Tensor};

/// Trait for Recurrent Neural Networks.
#[allow(clippy::upper_case_acronyms)]
pub trait RNN {
    type State: Clone;

    /// Returns the direction of the RNN.
    fn direction(&self) -> Direction;

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
    fn seq(&self, input: &Tensor) -> Result<Vec<Self::State>> {
        let batch_dim = input.dim(0)?;
        let state = self.zero_state(batch_dim)?;
        self.seq_init(input, &state)
    }

    /// Applies multiple steps of the recurrent network.
    ///
    /// The input should have dimensions [batch_size, seq_len, features].
    fn seq_init(&self, input: &Tensor, init_state: &Self::State) -> Result<Vec<Self::State>> {
        let (_b_size, seq_len, _features) = input.dims3()?;
        let mut output = Vec::with_capacity(seq_len);
        for seq_index in 0..seq_len {
            let index = if self.direction() == Direction::Forward {
                seq_index
            } else {
                seq_len - seq_index - 1
            };
            let input = input.i((.., index, ..))?.contiguous()?;
            let state = if seq_index == 0 {
                self.step(&input, init_state)?
            } else {
                self.step(&input, &output[seq_index - 1])?
            };
            output.push(state);
        }
        if self.direction() == Direction::Backward {
            output.reverse();
        }
        Ok(output)
    }

    /// Converts a sequence of state to a tensor.
    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor>;

    /// Combines forward and backward states to a tensor.
    fn bidirectional_states_to_tensor(
        &self,
        forward_states: &[Self::State],
        backward_states: &[Self::State],
    ) -> Result<Tensor>;
}

/// The state for a LSTM network, this contains two tensors.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct LSTMState {
    pub h: Tensor,
    pub c: Tensor,
}

impl LSTMState {
    pub fn new(h: Tensor, c: Tensor) -> Self {
        LSTMState { h, c }
    }

    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }

    /// The cell state vector.
    pub fn c(&self) -> &Tensor {
        &self.c
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Direction {
    Forward,
    Backward,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct LSTMConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
    pub layer_idx: usize,
    pub direction: Direction,
}

impl Default for LSTMConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

impl LSTMConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

/// A Long Short-Term Memory (LSTM) layer.
///
/// <https://en.wikipedia.org/wiki/Long_short-term_memory>
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct LSTM {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: LSTMConfig,
    device: Device,
    dtype: DType,
}

impl LSTM {
    /// Creates a LSTM layer.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        config: LSTMConfig,
        vb: crate::VarBuilder,
    ) -> Result<Self> {
        let layer_idx = config.layer_idx;
        let direction_str = match config.direction {
            Direction::Forward => "",
            Direction::Backward => "_reverse",
        };
        let w_ih = vb.get_with_hints(
            (4 * hidden_dim, in_dim),
            &format!("weight_ih_l{layer_idx}{direction_str}"), // Only a single layer is supported.
            config.w_ih_init,
        )?;
        let w_hh = vb.get_with_hints(
            (4 * hidden_dim, hidden_dim),
            &format!("weight_hh_l{layer_idx}{direction_str}"), // Only a single layer is supported.
            config.w_hh_init,
        )?;
        let b_ih = match config.b_ih_init {
            Some(init) => Some(vb.get_with_hints(
                4 * hidden_dim,
                &format!("bias_ih_l{layer_idx}{direction_str}"),
                init,
            )?),
            None => None,
        };
        let b_hh = match config.b_hh_init {
            Some(init) => Some(vb.get_with_hints(
                4 * hidden_dim,
                &format!("bias_hh_l{layer_idx}{direction_str}"),
                init,
            )?),
            None => None,
        };
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_dim,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &LSTMConfig {
        &self.config
    }
}

/// Creates a LSTM layer.
pub fn lstm(
    in_dim: usize,
    hidden_dim: usize,
    config: LSTMConfig,
    vb: crate::VarBuilder,
) -> Result<LSTM> {
    LSTM::new(in_dim, hidden_dim, config, vb)
}

impl RNN for LSTM {
    type State = LSTMState;

    fn direction(&self) -> Direction {
        self.config.direction
    }

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let zeros =
            Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;
        Ok(Self::State {
            h: zeros.clone(),
            c: zeros.clone(),
        })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let w_ih = input.matmul(&self.w_ih.t()?)?;
        let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
        let w_ih = match &self.b_ih {
            None => w_ih,
            Some(b_ih) => w_ih.broadcast_add(b_ih)?,
        };
        let w_hh = match &self.b_hh {
            None => w_hh,
            Some(b_hh) => w_hh.broadcast_add(b_hh)?,
        };
        let chunks = (&w_ih + &w_hh)?.chunk(4, 1)?;
        let in_gate = crate::ops::sigmoid(&chunks[0])?;
        let forget_gate = crate::ops::sigmoid(&chunks[1])?;
        let cell_gate = chunks[2].tanh()?;
        let out_gate = crate::ops::sigmoid(&chunks[3])?;

        let next_c = ((forget_gate * &in_state.c)? + (in_gate * cell_gate)?)?;
        let next_h = (out_gate * next_c.tanh()?)?;
        Ok(LSTMState {
            c: next_c,
            h: next_h,
        })
    }

    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::stack(&states, 1)
    }

    fn bidirectional_states_to_tensor(
        &self,
        forward_states: &[Self::State],
        backward_states: &[Self::State],
    ) -> Result<Tensor> {
        let combine_states = forward_states
            .iter()
            .zip(backward_states.iter())
            .collect::<Vec<_>>();
        let mut states = Vec::with_capacity(combine_states.len());
        for (f, b) in combine_states {
            states.push(Tensor::cat(&[&f.h, &b.h], 1)?);
        }
        Tensor::stack(&states, 1)
    }
}

/// The state for a GRU network, this contains a single tensor.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub struct GRUState {
    pub h: Tensor,
}

impl GRUState {
    /// The hidden state vector, which is also the output of the LSTM.
    pub fn h(&self) -> &Tensor {
        &self.h
    }
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy)]
pub struct GRUConfig {
    pub w_ih_init: super::Init,
    pub w_hh_init: super::Init,
    pub b_ih_init: Option<super::Init>,
    pub b_hh_init: Option<super::Init>,
    pub layer_idx: usize,
    pub direction: Direction,
}

impl Default for GRUConfig {
    fn default() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: Some(super::Init::Const(0.)),
            b_hh_init: Some(super::Init::Const(0.)),
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

impl GRUConfig {
    pub fn default_no_bias() -> Self {
        Self {
            w_ih_init: super::init::DEFAULT_KAIMING_UNIFORM,
            w_hh_init: super::init::DEFAULT_KAIMING_UNIFORM,
            b_ih_init: None,
            b_hh_init: None,
            layer_idx: 0,
            direction: Direction::Forward,
        }
    }
}

/// A Gated Recurrent Unit (GRU) layer.
///
/// <https://en.wikipedia.org/wiki/Gated_recurrent_unit>
#[allow(clippy::upper_case_acronyms)]
#[derive(Clone, Debug)]
pub struct GRU {
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Option<Tensor>,
    b_hh: Option<Tensor>,
    hidden_dim: usize,
    config: GRUConfig,
    device: Device,
    dtype: DType,
}

impl GRU {
    /// Creates a GRU layer.
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        config: GRUConfig,
        vb: crate::VarBuilder,
    ) -> Result<Self> {
        let layer_idx = config.layer_idx;
        let direction_str = match config.direction {
            Direction::Forward => "",
            Direction::Backward => "_reverse",
        };
        let w_ih = vb.get_with_hints(
            (3 * hidden_dim, in_dim),
            &format!("weight_ih_l{layer_idx}{direction_str}"),
            config.w_ih_init,
        )?;
        let w_hh = vb.get_with_hints(
            (3 * hidden_dim, hidden_dim),
            &format!("weight_hh_l{layer_idx}{direction_str}"),
            config.w_hh_init,
        )?;
        let b_ih = match config.b_ih_init {
            Some(init) => Some(vb.get_with_hints(
                3 * hidden_dim,
                &format!("bias_ih_l{layer_idx}{direction_str}"),
                init,
            )?),
            None => None,
        };
        let b_hh = match config.b_hh_init {
            Some(init) => Some(vb.get_with_hints(
                3 * hidden_dim,
                &format!("bias_hh_l{layer_idx}{direction_str}"),
                init,
            )?),
            None => None,
        };
        Ok(Self {
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            hidden_dim,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn config(&self) -> &GRUConfig {
        &self.config
    }
}

pub fn gru(
    in_dim: usize,
    hidden_dim: usize,
    config: GRUConfig,
    vb: crate::VarBuilder,
) -> Result<GRU> {
    GRU::new(in_dim, hidden_dim, config, vb)
}

impl RNN for GRU {
    type State = GRUState;

    fn direction(&self) -> Direction {
        self.config.direction
    }

    fn zero_state(&self, batch_dim: usize) -> Result<Self::State> {
        let h =
            Tensor::zeros((batch_dim, self.hidden_dim), self.dtype, &self.device)?.contiguous()?;
        Ok(Self::State { h })
    }

    fn step(&self, input: &Tensor, in_state: &Self::State) -> Result<Self::State> {
        let w_ih = input.matmul(&self.w_ih.t()?)?;
        let w_hh = in_state.h.matmul(&self.w_hh.t()?)?;
        let w_ih = match &self.b_ih {
            None => w_ih,
            Some(b_ih) => w_ih.broadcast_add(b_ih)?,
        };
        let w_hh = match &self.b_hh {
            None => w_hh,
            Some(b_hh) => w_hh.broadcast_add(b_hh)?,
        };
        let chunks_ih = w_ih.chunk(3, 1)?;
        let chunks_hh = w_hh.chunk(3, 1)?;
        let r_gate = crate::ops::sigmoid(&(&chunks_ih[0] + &chunks_hh[0])?)?;
        let z_gate = crate::ops::sigmoid(&(&chunks_ih[1] + &chunks_hh[1])?)?;
        let n_gate = (&chunks_ih[2] + (r_gate * &chunks_hh[2])?)?.tanh();

        let next_h = ((&z_gate * &in_state.h)? - ((&z_gate - 1.)? * n_gate)?)?;
        Ok(GRUState { h: next_h })
    }

    fn states_to_tensor(&self, states: &[Self::State]) -> Result<Tensor> {
        let states = states.iter().map(|s| s.h.clone()).collect::<Vec<_>>();
        Tensor::stack(&states, 1)
    }

    fn bidirectional_states_to_tensor(
        &self,
        forward_states: &[Self::State],
        backward_states: &[Self::State],
    ) -> Result<Tensor> {
        let combine_states = forward_states
            .iter()
            .zip(backward_states.iter())
            .collect::<Vec<_>>();
        let mut states = Vec::with_capacity(combine_states.len());
        for (f, b) in combine_states {
            states.push(Tensor::cat(&[&f.h, &b.h], 1)?);
        }
        Tensor::stack(&states, 1)
    }
}
