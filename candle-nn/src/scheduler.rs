use candle::Result;

/// The interface LR Schedulers should implement.
pub trait LRScheduler<T> {
    /// Step the scheduler and return the new learning rate.
    fn step(&mut self, params: T) -> Result<f64>;

    /// Get the current learning rate.
    fn get_lr(&self) -> f64;
}

/// A learning rate scheduler that uses a function to determine the learning rate.
/// The function should take a parameter of type `T` and return a `f64`.
pub struct FnLRScheduler<T> {
    pub func: Box<dyn Fn(T) -> Result<f64>>,
    pub lr: f64,
}

impl<T> FnLRScheduler<T> {
    pub fn new(func: Box<dyn Fn(T) -> Result<f64>>) -> Self {
        Self { func, lr: 0.0 }
    }
}

impl<T> LRScheduler<T> for FnLRScheduler<T> {
    fn step(&mut self, params: T) -> Result<f64> {
        self.lr = (self.func)(params)?;
        Ok(self.lr)
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}

/// Decays the learning rate of each parameter group by gamma every step_size epochs.
// https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR
pub struct StepLR {
    step_size: usize,
    last_epoch: usize,
    gamma: f64,
    lr: f64,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f64, lr: f64) -> Self {
        Self {
            step_size,
            last_epoch: 0,
            gamma,
            lr,
        }
    }
}

impl LRScheduler<()> for StepLR {
    fn step(&mut self, _params: ()) -> Result<f64> {
        self.last_epoch += 1;
        if self.last_epoch % self.step_size == 0 {
            self.lr *= self.gamma;
        }
        Ok(self.lr)
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}
/// Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.
// https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR
pub struct MultiStepLR {
    millstones: Vec<usize>,
    gamma: f64,
    last_epoch: usize,
    lr: f64,
}

impl MultiStepLR {
    pub fn new(millstones: Vec<usize>, gamma: f64, lr: f64) -> Result<Self> {
        // Ensure millstones are sorted.
        if !millstones.is_sorted() {
            candle::bail!("millstones should be sorted")
        }

        Ok(Self {
            millstones,
            gamma,
            last_epoch: 0,
            lr,
        })
    }
}

impl LRScheduler<()> for MultiStepLR {
    fn step(&mut self, _params: ()) -> Result<f64> {
        self.last_epoch += 1;
        if let Some(step) = self.millstones.first() {
            if self.last_epoch == *step {
                self.millstones.remove(0);
                self.lr *= self.gamma;
            }
        }
        Ok(self.lr)
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}

/// Set the learning rate of each parameter group using a cosine annealing schedule.
//https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR
pub struct CosineAnnealingLR {
    t_max: usize,
    last_epoch: usize,
    eta_min: f64,
    lr: f64,
}

impl CosineAnnealingLR {
    pub fn new(t_max: usize, eta_min: f64, lr: f64) -> Self {
        Self {
            t_max,
            last_epoch: 0,
            eta_min,
            lr,
        }
    }
}

impl LRScheduler<()> for CosineAnnealingLR {
    fn step(&mut self, _params: ()) -> Result<f64> {
        self.lr = self.eta_min
            + 0.5
                * (self.lr - self.eta_min)
                * (1. + ((self.last_epoch as f64 / self.t_max as f64) * std::f64::consts::PI)).cos();
        self.last_epoch += 1;
        self.last_epoch = self.last_epoch.min(self.t_max);
        Ok(self.lr)
    }

    fn get_lr(&self) -> f64 {
        self.lr
    }
}
