//! Logit Processing and Sampling
//!
//! Functionality for modeling sampling strategies and logits processing in text generation
//! with support for temperature-based sampling, top-k filtering, nucleus sampling (top-p),
//! and combinations thereof.
use candle::{Context, DType, Error, Result, Tensor};
use rand::{distributions::Distribution, SeedableRng};
use std::sync::{Arc, Mutex};
#[derive(Clone, PartialEq, Debug)]
pub enum Sampling {
    ArgMax,
    All { temperature: f64 },
    TopK { k: usize, temperature: f64 },
    TopP { p: f64, temperature: f64 },
    TopKThenTopP { k: usize, p: f64, temperature: f64 },
}

pub struct LogitsProcessor {
    rng: Arc<Mutex<rand::rngs::StdRng>>,
    sampling: Sampling,
}

impl LogitsProcessor {
    pub fn from_sampling(seed: u64, sampling: Sampling) -> Self {
        let rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            rng: Arc::new(Mutex::new(rng)),
            sampling,
        }
    }

    pub fn new(seed: u64, temperature: Option<f64>, top_p: Option<f64>) -> Self {
        let temperature = temperature.and_then(|v| if v < 1e-7 { None } else { Some(v) });
        let sampling = match temperature {
            None => Sampling::ArgMax,
            Some(temperature) => match top_p {
                None => Sampling::All { temperature },
                Some(p) => Sampling::TopP { p, temperature },
            },
        };
        Self::from_sampling(seed, sampling)
    }

    fn sample_argmax(&mut self, logits: Tensor) -> Result<u32> {
        let logits_v: Vec<f32> = logits.to_vec1()?;
        let next_token = logits_v
            .iter()
            .enumerate()
            .max_by(|(_, u), (_, v)| u.total_cmp(v))
            .map(|(i, _)| i as u32)
            .context("empty logits")?;
        Ok(next_token)
    }

    fn sample_multinomial(&self, prs: &Vec<f32>) -> Result<u32> {
        let distr = rand::distributions::WeightedIndex::new(prs).map_err(Error::wrap)?;
        let mut rng = self.rng.lock().unwrap();
        let next_token = distr.sample(&mut *rng) as u32;
        Ok(next_token)
    }

    /// top-p sampling (or "nucleus sampling") samples from the smallest set of tokens that exceed
    /// probability top_p. This way we never sample tokens that have very low probabilities and are
    /// less likely to go "off the rails".
    fn sample_topp(&self, logits: &Tensor, top_p: f32) -> Result<u32> {
        let mut prs: Vec<f32> = logits.to_vec1()?;
        let argsort_indices: Vec<u32> = logits.arg_sort_last_dim(false)?.to_vec1()?;
        // // Slower approach on CPU.
        // let mut argsort_indices = (0..prs.len()).collect::<Vec<_>>();
        // argsort_indices.sort_by(|&i, &j| prs[j].total_cmp(&prs[i]));
        // Clamp smaller probabilities to zero.
        let mut cumsum = 0.;
        for index in &argsort_indices {
            if cumsum >= top_p {
                prs[*index as usize] = 0.0;
            } else {
                cumsum += prs[*index as usize];
            }
        }
        // Sample with clamped probabilities.
        self.sample_multinomial(&prs)
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    fn sample_topk(&self, logits: &Tensor, top_k: usize) -> Result<u32> {
        if top_k >= *logits.layout().dims().last().unwrap() {
            let (sorted, _) = logits.sort_last_dim(false)?;
            let prs: Vec<f32> = sorted.to_vec1()?;
            self.sample_multinomial(&prs)
        } else {
            let (sorted, asort) = logits.sort_last_dim(false)?;
            let prs: Vec<f32> = sorted.to_vec1()?[0..top_k].to_vec();
            let index = self.sample_multinomial(&prs)?;
            let indices: Vec<u32> = asort.to_vec1()?[0..top_k].to_vec();
            Ok(indices[index as usize] as u32)
        }
    }

    // top-k sampling samples from the k tokens with the largest probabilities.
    // then top-p sampling.
    fn sample_topk_topp(&self, logits: &Tensor, top_k: usize, top_p: f32) -> Result<u32> {
        if top_k >= *logits.layout().dims().last().unwrap() {
            let (sorted, _) = logits.sort_last_dim(false)?;
            let prs: Vec<f32> = sorted.to_vec1()?;
            self.sample_multinomial(&prs)
        } else {
            let (sorted, asort) = logits.sort_last_dim(false)?;
            let indices: Vec<u32> = asort.to_vec1()?[0..top_k].to_vec();
            let mut prs: Vec<f32> = sorted.to_vec1()?[0..top_k].to_vec();
            let sum_p = prs.iter().sum::<f32>();
            let index = if top_p <= 0.0 || top_p >= sum_p {
                self.sample_multinomial(&prs)?
            } else {
                let mut cumsum = 0.;
                for i in 0..prs.len() {
                    if cumsum >= top_p {
                        prs[i] = 0.0;
                    } else {
                        cumsum += prs[i];
                    }
                }
                // Sample with clamped probabilities.
                self.sample_multinomial(&prs)?
            };
            Ok(indices[index as usize] as u32)
        }
    }

    pub fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        self.sample_f(logits)
    }

    pub fn sample_f(&mut self, logits: &Tensor) -> Result<u32> {
        let logits = logits.to_dtype(DType::F32)?;
        let prs = |temperature: f64| -> Result<Tensor> {
            let logits = (&logits / temperature)?;
            let prs = candle_nn::ops::softmax_last_dim(&logits)?;
            Ok(prs)
        };

        let next_token = match &self.sampling {
            Sampling::ArgMax => self.sample_argmax(logits)?,
            Sampling::All { temperature } => {
                let prs = prs(*temperature)?.to_vec1()?;
                self.sample_multinomial(&prs)?
            }
            Sampling::TopP { p, temperature } => {
                let prs = prs(*temperature)?;
                if *p <= 0.0 || *p >= 1.0 {
                    // simply sample from the predicted probability distribution
                    self.sample_multinomial(&prs.to_vec1()?)?
                } else {
                    // top-p (nucleus) sampling, clamping the least likely tokens to zero
                    self.sample_topp(&prs, *p as f32)?
                }
            }
            Sampling::TopK { k, temperature } => {
                let prs = prs(*temperature)?;
                self.sample_topk(&prs, *k)?
            }
            Sampling::TopKThenTopP { k, p, temperature } => {
                let prs = prs(*temperature)?;
                self.sample_topk_topp(&prs, *k, *p as f32)?
            }
        };
        Ok(next_token)
    }
}
