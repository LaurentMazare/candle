use candle::{Device, Result, Tensor};

pub fn get_noise(
    num_samples: usize,
    height: usize,
    width: usize,
    device: &Device,
) -> Result<Tensor> {
    Tensor::randn(0f32, 1., (num_samples, 16, height, width), device)
}

pub struct State {
    pub img: Tensor,
    pub img_ids: Tensor,
    pub txt: Tensor,
    pub txt_ids: Tensor,
    pub vec: Tensor,
}

impl State {
    pub fn new(t5_emb: &Tensor, clip_emb: &Tensor, img: &Tensor) -> Result<Self> {
        let (bs, c, h, w) = img.dims4()?;
        let dev = img.device();
        let img = img.reshape((bs, c, h / 2, 2, w / 2, 2))?; // (b, c, h, ph, w, pw)
        let img = img.permute((0, 2, 4, 1, 3, 5))?; // (b, h, w, c, ph, pw)
        let img = img.reshape((bs, h / 2 * w / 2, c * 4))?;
        let img_ids = Tensor::stack(
            &[
                Tensor::full(0u32, (h / 2, w / 2), dev)?,
                Tensor::arange(0u32, h as u32 / 2, dev)?
                    .reshape(((), 1))?
                    .broadcast_as((h / 2, w / 2))?,
                Tensor::arange(0u32, w as u32 / 2, dev)?
                    .reshape((1, ()))?
                    .broadcast_as((h / 2, w / 2))?,
            ],
            2,
        )?;
        let img_ids = img_ids.reshape((1, h / 2 * w / 2, 3))?;
        let img_ids = img_ids.repeat((bs, 1, 1))?;
        let txt = t5_emb.repeat(bs)?;
        let txt_ids = Tensor::zeros((bs, txt.dim(1)?, 3), candle::DType::F32, dev)?;
        let vec = clip_emb.repeat(bs)?;
        Ok(Self {
            img,
            img_ids,
            txt,
            txt_ids,
            vec,
        })
    }
}

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1. / t - 1.).powf(sigma))
}

/// `shift` is a triple `(image_seq_len, base_shift, max_shift)`.
pub fn get_schedule(num_steps: usize, shift: Option<(usize, f64, f64)>) -> Vec<f64> {
    let timesteps: Vec<f64> = (0..=num_steps)
        .map(|v| v as f64 / num_steps as f64)
        .collect();
    match shift {
        None => timesteps,
        Some((image_seq_len, y1, y2)) => {
            let (x1, x2) = (256., 4096.);
            let m = (y2 - y1) / (x2 - x1);
            let b = y1 - m * x1;
            let mu = m * image_seq_len as f64 + b;
            timesteps
                .into_iter()
                .map(|v| time_shift(mu, 1., v))
                .collect()
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn denoise(
    model: &super::model::Flux,
    img: &Tensor,
    img_ids: &Tensor,
    txt: &Tensor,
    txt_ids: &Tensor,
    vec_: &Tensor,
    timesteps: &[f64],
    guidance: f64,
) -> Result<Tensor> {
    let b_sz = img.dim(0)?;
    let dev = img.device();
    let guidance = Tensor::full(guidance as f32, b_sz, dev)?;
    let mut img = img.clone();
    for window in timesteps.windows(2) {
        let (t_prev, t_curr) = match window {
            [a, b] => (a, b),
            _ => continue,
        };
        let t_vec = Tensor::full(*t_curr as f32, b_sz, dev)?;
        let pred = model.forward(&img, img_ids, txt, txt_ids, &t_vec, vec_, Some(&guidance))?;
        img = (img + pred * (t_prev - t_curr))?
    }
    Ok(img)
}
