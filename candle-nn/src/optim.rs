//! Various optimization algorithms.
//!
//! This contains to major traits: `Optimizer` and `Scheduler`. The `Optimizer` is designed to contain
//! the `Scheduler`, and handles the process of actually doing the optimization. In contrast,
//! the `Scheduler` purely focuses on the learning rate.

use std::{
    fmt::Debug,
    sync::{Arc, Mutex},
};

use candle::{Result, Tensor, Var};

/// The trait optimizers should implement.
pub trait Optimizer: Sized {
    type SchedulerOutput: Sized;
    type Config: Sized;

    fn new(
        vars: Vec<Var>,
        scheduler: Arc<Mutex<dyn Scheduler<Output = Self::SchedulerOutput>>>,
        config: Self::Config,
    ) -> Result<Self>;

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()>;

    fn empty(
        scheduler: Arc<Mutex<dyn Scheduler<Output = Self::SchedulerOutput>>>,
        config: Self::Config,
    ) -> Result<Self> {
        Self::new(vec![], scheduler, config)
    }

    fn backward_step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    fn from_slice(
        vars: &[&Var],
        scheduler: Arc<Mutex<dyn Scheduler<Output = Self::SchedulerOutput>>>,
        config: Self::Config,
    ) -> Result<Self> {
        let vars: Vec<_> = vars.iter().map(|&v| v.clone()).collect();
        Self::new(vars, scheduler, config)
    }
}

/// A trait to abstract schedulers.
pub trait Scheduler: Debug {
    type Output: Sized;

    fn step(&mut self) -> Result<Self::Output>;
}

/// A trait to simplify construction of schedulers.
pub trait SchedulerCreator: Debug {
    type Config: Sized;
    type Output: Sized;

    fn new(config: Self::Config) -> Result<Arc<Mutex<dyn Scheduler<Output = Self::Output>>>>;
}

macro_rules! get_scheduler {
    ($thing:expr) => {
        loop {
            if let Ok(inner) = $thing.try_lock() {
                break inner;
            }
        }
    };
}

/// A scheduler which maintains a static learning rate.
#[derive(Debug)]
pub struct ConstantScheduler {
    lr: f64,
}

impl Scheduler for ConstantScheduler {
    type Output = f64;

    fn step(&mut self) -> Result<f64> {
        Ok(self.lr)
    }
}

impl SchedulerCreator for ConstantScheduler {
    type Config = f64;
    type Output = f64;

    fn new(lr: Self::Config) -> Result<Arc<Mutex<dyn Scheduler<Output = Self::Output>>>> {
        Ok(Arc::new(Mutex::new(Self { lr })))
    }
}

/// Optimizer for Stochastic Gradient Descent.
///
/// Contrary to the PyTorch implementation of SGD, this version does not support momentum.
#[derive(Debug)]
pub struct SGD {
    vars: Vec<Var>,
    scheduler: Arc<Mutex<dyn Scheduler<Output = f64>>>,
}

impl Optimizer for SGD {
    type SchedulerOutput = f64;
    type Config = ();

    fn new(
        vars: Vec<Var>,
        scheduler: Arc<Mutex<dyn Scheduler<Output = f64>>>,
        _: (),
    ) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .collect();
        Ok(Self { vars, scheduler })
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        let mut scheduler = get_scheduler!(self.scheduler);
        let lr = scheduler.step()?;
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * lr)?)?)?;
            }
        }
        Ok(())
    }
}

impl SGD {
    pub fn into_inner(self) -> Vec<Var> {
        self.vars
    }

    pub fn push(&mut self, var: &Var) {
        self.vars.push(var.clone())
    }
}

/// Parameters for the AdamW scheduler.
#[derive(Clone, Debug)]
pub struct SchedulerParamsAdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
}

impl Default for SchedulerParamsAdamW {
    fn default() -> Self {
        Self {
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            weight_decay: 0.01,
        }
    }
}

/// Parameters for the AdamW optimizer.
#[derive(Clone, Debug)]
pub struct ParamsAdamW {
    pub eps: f64,
}

impl Default for ParamsAdamW {
    fn default() -> Self {
        Self { eps: 1e-8 }
    }
}

/// A scheduler for AdamW.
#[derive(Debug)]
pub struct AdamWScheduler {
    params: SchedulerParamsAdamW,
}

#[derive(Debug)]
pub struct AdamWSchedulerOutput {
    pub lr: f64,
    pub lr_lambda: f64,
    pub beta1: f64,
    pub beta2: f64,
}

impl Scheduler for AdamWScheduler {
    type Output = AdamWSchedulerOutput;

    fn step(&mut self) -> Result<AdamWSchedulerOutput> {
        let lr = self.params.lr;
        let lambda = self.params.weight_decay;
        let lr_lambda = lr * lambda;
        let beta1 = self.params.beta1;
        let beta2 = self.params.beta2;
        Ok(AdamWSchedulerOutput {
            lr,
            lr_lambda,
            beta1,
            beta2,
        })
    }
}

impl SchedulerCreator for AdamWScheduler {
    type Config = SchedulerParamsAdamW;
    type Output = AdamWSchedulerOutput;

    fn new(params: Self::Config) -> Result<Arc<Mutex<dyn Scheduler<Output = Self::Output>>>> {
        Ok(Arc::new(Mutex::new(Self { params })))
    }
}

#[derive(Debug)]
struct VarAdamW {
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

#[derive(Debug)]
pub struct AdamW {
    vars: Vec<VarAdamW>,
    step_t: usize,
    scheduler: Arc<Mutex<dyn Scheduler<Output = AdamWSchedulerOutput>>>,
    params: ParamsAdamW,
}

impl Optimizer for AdamW {
    type SchedulerOutput = AdamWSchedulerOutput;
    type Config = ParamsAdamW;

    fn new(
        vars: Vec<Var>,
        scheduler: Arc<Mutex<dyn Scheduler<Output = AdamWSchedulerOutput>>>,
        params: ParamsAdamW,
    ) -> Result<Self> {
        let vars = vars
            .into_iter()
            .filter(|var| var.dtype().is_float())
            .map(|var| {
                let dtype = var.dtype();
                let shape = var.shape();
                let device = var.device();
                let first_moment = Var::zeros(shape, dtype, device)?;
                let second_moment = Var::zeros(shape, dtype, device)?;
                Ok(VarAdamW {
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            scheduler,
            params,
        })
    }

    fn step(&mut self, grads: &candle::backprop::GradStore) -> Result<()> {
        self.step_t += 1;
        let AdamWSchedulerOutput {
            lr,
            lr_lambda,
            beta1,
            beta2,
        } = get_scheduler!(self.scheduler).step()?;
        let scale_m = 1f64 / (1f64 - beta1.powi(self.step_t as i32));
        let scale_v = 1f64 / (1f64 - beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let theta = &var.var;
            let m = &var.first_moment;
            let v = &var.second_moment;
            if let Some(g) = grads.get(theta) {
                // This involves locking 3 RWLocks per params, if the parameters are large this
                // should not be an issue but this may be problematic with models with lots of
                // small parameters.
                let next_m = ((m.as_tensor() * beta1)? + (g * (1.0 - beta1))?)?;
                let next_v = ((v.as_tensor() * beta2)? + (g.sqr()? * (1.0 - beta2))?)?;
                let m_hat = (&next_m * scale_m)?;
                let v_hat = (&next_v * scale_v)?;
                let next_theta = (theta.as_tensor() * (1f64 - lr_lambda))?;
                let adjusted_grad = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
                let next_theta = (next_theta - (adjusted_grad * lr)?)?;
                m.set(&next_m)?;
                v.set(&next_v)?;
                theta.set(&next_theta)?;
            }
        }
        Ok(())
    }
}

impl AdamW {
    pub fn new_lr(vars: Vec<Var>, learning_rate: f64) -> Result<Self> {
        let params = ParamsAdamW::default();
        let scheduler_params = SchedulerParamsAdamW {
            lr: learning_rate,
            ..SchedulerParamsAdamW::default()
        };
        Self::new(
            vars,
            Arc::new(Mutex::new(AdamWScheduler {
                params: scheduler_params,
            })),
            params,
        )
    }

    pub fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }
}
