use crate::config::ImagePreProcessorConfig; // Use the same type from config
use candle::{DType, Device, Result, Tensor};
use image::{imageops::overlay, DynamicImage, GenericImageView, Rgb, RgbImage};

use core::num;
use std::collections::HashMap;

pub struct ImageProcessor {
    pub do_convert_rgb: bool,
    pub do_resize: bool,
    pub size: Option<HashMap<String, usize>>,
    pub resample: usize,
    pub do_rescale: bool,
    pub rescale_factor: f32,
    pub do_normalize: bool,
    pub image_mean: Option<Vec<f32>>,
    pub image_std: Option<Vec<f32>>,
    pub do_pad: bool,
    pub max_image_tiles: usize,
}
impl ImageProcessor {
    pub fn new(cfg: &ImagePreProcessorConfig) -> Self {
        let do_convert_rgb = cfg.do_convert_rgb;
        let do_resize = cfg.do_resize;
        let size = Some(cfg.size.clone());
        let resample = cfg.resample;
        let do_rescale = cfg.do_rescale;
        let rescale_factor = cfg.rescale_factor;
        let do_normalize = cfg.do_normalize;
        let image_mean = Some(cfg.image_mean.clone());
        let image_std = Some(cfg.image_std.clone());
        let do_pad = cfg.do_pad;
        let max_image_tiles = cfg.max_image_tiles;
        Self {
            do_convert_rgb,
            do_resize,
            size,
            resample,
            do_rescale,
            rescale_factor,
            do_normalize,
            image_mean,
            image_std,
            do_pad,
            max_image_tiles,
        }
    }

    pub fn convert_rgb(&self, image: &DynamicImage) -> DynamicImage {
        // Do we need this function at all?????? we do need it in python because of PIL
        // but maybe not here!!!!
        image.clone()
    }

    pub fn to_tensor(&self, image: &DynamicImage) -> Result<Tensor> {
        let img = image.to_rgb8().into_raw();
        let (width, height) = image.dimensions();
        // only for internal compute
        Tensor::from_vec(img, (height as usize, width as usize, 3), &Device::Cpu)?
            .to_dtype(DType::F32)
    }

    pub fn to_channel_dimension_format(&self, tensor: &Tensor) -> Result<Tensor> {
        tensor.permute((2, 0, 1))
    }

    pub fn get_all_supported_aspect_ratios(&self, max_image_tiles: usize) -> Vec<(usize, usize)> {
        // Generate all possible arrangements
        let possible_canvas_sizes: Vec<(usize, usize)> = (1..=max_image_tiles)
            .flat_map(|width| {
                (1..=max_image_tiles)
                    .filter(move |height| width * height <= max_image_tiles)
                    .map(move |height| (height, width))
            })
            .collect();

        possible_canvas_sizes
    }

    pub fn get_optimal_tiled_canvas(
        &self,
        image_height: usize,
        image_width: usize,
        max_image_tiles: usize,
        tile_size: usize,
    ) -> (usize, usize) {
        let possible_canvas_sizes = self.get_all_supported_aspect_ratios(max_image_tiles);
        let possible_canvas_sizes: Vec<(usize, usize)> = possible_canvas_sizes
            .iter()
            .map(|(h, w)| (h * tile_size, w * tile_size))
            .collect();

        // Calculate scales for each arrangement
        let scales: Vec<f32> = possible_canvas_sizes
            .iter()
            .map(|(h, w)| {
                let scale_h = *h as f32 / image_height as f32;
                let scale_w = *w as f32 / image_width as f32;
                scale_h.min(scale_w)
            })
            .collect();

        // Find the optimal scale
        let selected_scale = scales
            .iter()
            .copied()
            .filter(|x| *x >= 1.0)
            .min_by(|x, y| x.partial_cmp(y).unwrap())
            .unwrap_or_else(|| {
                scales
                    .iter()
                    .copied()
                    .max_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap()
            });

        // Find arrangement with minimum area for selected scale
        let (optimal_height, optimal_width) = possible_canvas_sizes
            .iter()
            .zip(scales.iter())
            .filter(|(_, &scale)| scale == selected_scale)
            .min_by_key(|((h, w), _)| h * w)
            .map(|((h, w), _)| (*h, *w))
            .unwrap();

        (optimal_height, optimal_width)
    }

    pub fn get_image_size_fit_to_canvas(
        &self,
        image_height: usize,
        image_width: usize,
        canvas_height: usize,
        canvas_width: usize,
        tile_size: usize,
    ) -> (usize, usize) {
        // Set target image size in between `tile_size` and canvas_size
        let target_width = match (image_width <= tile_size, image_width >= canvas_width) {
            (true, _) => tile_size,
            (_, true) => canvas_width,
            _ => image_width,
        };
        let target_height = match (image_height <= tile_size, image_height >= canvas_height) {
            (true, _) => tile_size,
            (_, true) => canvas_height,
            _ => image_height,
        };

        let scale_h = target_height as f32 / image_height as f32;
        let scale_w = target_width as f32 / image_width as f32;

        let (new_height, new_width) = if scale_w < scale_h {
            let h = std::cmp::min((scale_w * image_height as f32) as usize, target_height);
            (h, target_width)
        } else {
            let w = std::cmp::min((scale_h * image_width as f32) as usize, target_width);
            (target_height, w)
        };

        (new_height, new_width)
    }

    pub fn resize(
        &self,
        image: &DynamicImage,
        size: &HashMap<String, usize>,
        resample: usize,
        max_image_tiles: usize,
    ) -> (DynamicImage, (usize, usize)) {
        let image_height = image.height() as usize;
        let image_width = image.width() as usize;

        let tile_size = size.get("height").unwrap();
        let (canvas_height, canvas_width) =
            self.get_optimal_tiled_canvas(image_height, image_width, max_image_tiles, *tile_size);

        let num_tiles_height = canvas_height / tile_size;
        let num_tiles_width = canvas_width / tile_size;

        let (new_height, new_width) = self.get_image_size_fit_to_canvas(
            image_height,
            image_width,
            canvas_height,
            canvas_width,
            *tile_size,
        );

        let new_image = image.resize(
            new_width as u32,
            new_height as u32,
            image::imageops::FilterType::CatmullRom,
        );

        (new_image, (num_tiles_height, num_tiles_width))
    }

    pub fn pad(
        &self,
        image: &DynamicImage,
        size: &HashMap<String, usize>,
        aspect_ratio: (usize, usize),
    ) -> DynamicImage {
        let size_h = size.get("height").unwrap();
        let size_w = size.get("width").unwrap();

        let (num_tiles_height, num_tiles_width) = aspect_ratio;
        let padded_height = num_tiles_height * size_h;
        let padded_width = num_tiles_width * size_w;

        let mut new_image = DynamicImage::from(RgbImage::from_pixel(
            padded_width as u32,
            padded_height as u32,
            Rgb::from([0, 0, 0]),
        ));
        // Calculate center position
        // let x = ((padded_width - image.width() as usize) / 2) as i64;
        // let y = ((padded_height - image.height() as usize) / 2) as i64;
        overlay(&mut new_image, image, 0, 0);
        new_image
    }

    pub fn rescale(&self, tensor: &Tensor) -> Result<Tensor> {
        let rescale_factor = self.rescale_factor as f64;
        tensor.affine(rescale_factor, 0.0)
    }

    pub fn normalize(&self, tensor: &Tensor) -> Result<Tensor> {
        let image_mean = self.image_mean.clone().unwrap();
        let image_std = self.image_std.clone().unwrap();
        let mean = Tensor::from_vec(image_mean, (3, 1, 1), &Device::Cpu)?;
        let std = Tensor::from_vec(image_std, (3, 1, 1), &Device::Cpu)?;
        // println!("{:?} {:?} {:?}", tensor.shape(), mean.shape(), std.shape());
        // mean.broadcast_as(shape)
        tensor.broadcast_sub(&mean)?.broadcast_div(&std)
    }

    pub fn split_to_tiles(
        &self,
        tensor: &Tensor,
        num_tiles_height: usize,
        num_tiles_width: usize,
    ) -> Tensor {
        let tensor_dim = tensor.dims();
        let num_channels = tensor_dim[0];
        let height = tensor_dim[1];
        let width = tensor_dim[2];

        let tile_height = height / num_tiles_height;
        let tile_width = width / num_tiles_width;

        let mut tensor = tensor
            .reshape((
                num_channels,
                num_tiles_height,
                tile_height,
                num_tiles_width,
                tile_width,
            ))
            .unwrap();

        // Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
        tensor = tensor.permute((1, 3, 0, 2, 4)).unwrap();

        // # Reshape into the desired output shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
        tensor = tensor
            .reshape((
                num_tiles_width * num_tiles_height,
                num_channels,
                tile_height,
                tile_width,
            ))
            .unwrap();

        tensor
    }

    pub fn convert_aspect_ratios_to_ids(
        &self,
        aspect_ratio: (usize, usize),
        max_image_tiles: usize,
    ) -> usize {
        let supported_aspect_ratios = self.get_all_supported_aspect_ratios(max_image_tiles);

        let aspect_ratios_ids = supported_aspect_ratios
            .iter()
            .position(|element| *element == aspect_ratio)
            .unwrap();

        aspect_ratios_ids + 1
    }

    pub fn build_aspect_ratio_mask(
        &self,
        aspect_ratio: (usize, usize),
        max_image_tiles: usize,
    ) -> Result<Tensor> {
        let mut aspect_ratio_mask = vec![0 as f32; max_image_tiles];
        for i in 0..(aspect_ratio.0 * aspect_ratio.1) {
            aspect_ratio_mask[i] = 1.0;
        }

        Tensor::from_vec(aspect_ratio_mask, max_image_tiles, &Device::Cpu)
    }

    pub fn preprocess(
        &self,
        image: &DynamicImage,
    ) -> Result<(Tensor, Tensor, Tensor, Vec<Vec<usize>>)> {
        let image = if self.do_convert_rgb {
            self.convert_rgb(image)
        } else {
            image.clone()
        };

        // do_resize=False is not supported, validated
        let (image, aspect_ratio) = self.resize(
            &image,
            &self.size.as_ref().unwrap(),
            self.resample,
            self.max_image_tiles,
        );

        // do_pad=False is not supported, validated
        let image = self.pad(&image, &self.size.as_ref().unwrap(), aspect_ratio);

        // let _ = image.save("after-resize.png");

        let image = self.to_tensor(&image)?;
        let mut image = self.to_channel_dimension_format(&image)?;

        if self.do_rescale {
            image = self.rescale(&image)?;
        }

        if self.do_normalize {
            image = self.normalize(&image)?;
        }

        image.save_safetensors("normalized", "./after-norm-r")?;

        let (num_tiles_height, num_tiles_width) = aspect_ratio;
        image = self.split_to_tiles(&image, num_tiles_height, num_tiles_width);

        let aspect_ratio_ids =
            self.convert_aspect_ratios_to_ids(aspect_ratio, self.max_image_tiles);
        let aspect_ratio_mask = self.build_aspect_ratio_mask(aspect_ratio, self.max_image_tiles)?;

        // TODO: we need to add batch processing
        // for now just reshaping tensors to make it same as
        // python impelementation
        image = image.reshape((
            1,               // batch_size
            1,               // num input images for the prompt
            image.dims()[0], // num tiles
            image.dims()[1], // channel
            image.dims()[2], // h
            image.dims()[3], // w
        ))?;
        let aspect_ratio_ids =
            Tensor::from_vec(vec![aspect_ratio_ids as u32], (1, 1), &Device::Cpu)?;
        let aspect_ratio_mask = aspect_ratio_mask.reshape((1, 1, aspect_ratio_mask.dims()[0]))?;
        let num_tiles = vec![vec![image.dims()[2]]];
        // println!(
        //     "{:?} {:?} {:?} {:?}",
        //     image.shape(),
        //     aspect_ratio_ids.shape(),
        //     aspect_ratio_mask.shape(),
        //     num_tiles
        // );
        let mut x = Tensor::zeros((5, 5), DType::F32, &Device::Cpu)?;

        Ok((image, aspect_ratio_ids, aspect_ratio_mask, num_tiles))
    }
}
