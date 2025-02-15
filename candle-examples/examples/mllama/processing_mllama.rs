use anyhow::{bail, Error as E, Result};
use candle::{DType, Device, Tensor};
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::image_processor::ImageProcessor;

pub struct MllamaProcessorOutput {
    pub input_ids: Tensor,
    pub attention_mask: Tensor,
    pub pixel_values: Tensor,
    pub aspect_ratio_ids: Tensor,
    pub aspect_ratio_mask: Tensor,
    pub cross_attention_mask: Tensor,
}

pub struct MllamaProcessor {
    pub image_token: String,
    pub image_token_id: u32,
    pub python_token: String,
    pub python_token_id: u32,
    pub bos_token: String,
    // pub chat_template: String,
    pub tokenizer: Tokenizer,
    pub image_processor: ImageProcessor,
}
impl MllamaProcessor {
    pub fn new(image_processor: ImageProcessor, tokenizer: Tokenizer) -> Self {
        let image_token = String::from("<|image|>");
        let image_token_id = tokenizer.token_to_id(image_token.as_str()).unwrap();

        let python_token = String::from("<|python_tag|>");
        let python_token_id = tokenizer.token_to_id(&python_token.as_str()).unwrap();

        let bos_token = String::from("<|begin_of_text|>");

        Self {
            image_token,
            image_token_id,
            python_token,
            python_token_id,
            bos_token,
            tokenizer,
            image_processor,
        }
    }

    pub fn build_string_from_input(
        &self,
        prompt: String,
        bos_token: String,
        image_token: String,
    ) -> String {
        if let Some(_) = prompt.find(&bos_token) {
            return prompt.to_string();
        }
        let mut prompt = prompt.clone();
        let mut num_image_tokens_on_start = 0;
        while prompt.starts_with(&image_token) {
            prompt = prompt[image_token.len()..].to_string();
            num_image_tokens_on_start += 1;
        }

        let mut result = String::new();
        for _ in 0..num_image_tokens_on_start {
            result.push_str(&image_token);
        }
        result.push_str(&bos_token);
        result.push_str(&prompt);
        result
    }

    pub fn get_cross_attention_token_mask(
        &self,
        input_ids: &Vec<u32>,
        image_token_id: u32,
    ) -> Vec<Vec<i32>> {
        let image_token_locations: Vec<i32> = input_ids
            .iter()
            .enumerate()
            .filter(|(_, &token)| token == image_token_id)
            .map(|(i, _)| i as i32)
            .collect();

        if image_token_locations.len() == 0 {
            return Vec::<Vec<i32>>::new();
        }

        // only one image present, unmask until end of sequence
        if image_token_locations.len() == 1 {
            return vec![vec![image_token_locations[0], -1]];
        }

        let mut vision_mask: Vec<Vec<i32>> = (1..image_token_locations.len())
            .map(|i| vec![image_token_locations[i - 1], image_token_locations[i]])
            .collect();

        vision_mask.push(vec![
            image_token_locations[image_token_locations.len() - 1],
            input_ids.len() as i32,
        ]);

        let mut last_mask_end = *vision_mask.last().unwrap().last().unwrap();
        for i in (0..vision_mask.len()).rev() {
            if vision_mask[i][0] == vision_mask[i][1] - 1 {
                vision_mask[i][1] = last_mask_end;
            }
            last_mask_end = vision_mask[i][1];
        }

        vision_mask
    }

    pub fn convert_sparse_cross_attention_mask_to_dense(
        &self,
        cross_attention_token_mask: Vec<Vec<Vec<i32>>>,
        num_tiles: Vec<Vec<usize>>,
        max_num_tiles: usize,
        length: usize,
    ) -> candle::Result<Tensor> {
        let batch_size = cross_attention_token_mask.len();

        let max_num_images = cross_attention_token_mask
            .iter()
            .map(|x| x.len())
            .max()
            .unwrap_or(0);

        let mut cross_attention_mask = Tensor::zeros(
            (batch_size, length, max_num_images, max_num_tiles),
            DType::F32,
            &Device::Cpu,
        )?;

        for (sample_idx, (sample_masks, sample_num_tiles)) in cross_attention_token_mask
            .iter()
            .zip(num_tiles.iter())
            .enumerate()
        {
            for (mask_idx, (locations, mask_num_tiles)) in
                sample_masks.iter().zip(sample_num_tiles).enumerate()
            {
                if locations.len() == 2 {
                    let start = locations[0];
                    let mut end = locations[1];
                    end = std::cmp::max(end, length as i32);
                    if end == -1 {
                        end = length as i32;
                    }
                    let ones = Tensor::ones(
                        (
                            1 as usize,
                            (end - start) as usize,
                            1 as usize,
                            *mask_num_tiles as usize,
                        ),
                        DType::F32,
                        &Device::Cpu,
                    )?;
                    cross_attention_mask.slice_assign(
                        &[
                            sample_idx..=sample_idx,
                            (start as usize)..=(end as usize - 1),
                            mask_idx..=mask_idx,
                            0..=(*mask_num_tiles as usize - 1),
                        ],
                        &ones,
                    )?;
                }
            }
        }

        Ok(cross_attention_mask)
    }

    pub fn process(&self, image: DynamicImage, prompt: String) -> Result<MllamaProcessorOutput> {
        let text =
            self.build_string_from_input(prompt, self.bos_token.clone(), self.image_token.clone());

        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let (image, aspect_ratio_ids, aspect_ratio_mask, num_tiles) =
            self.image_processor.preprocess(&image)?;

        let mut cross_attention_token_mask = Vec::new();
        cross_attention_token_mask
            .push(self.get_cross_attention_token_mask(&encoding, self.image_token_id));

        let cross_attention_mask = self.convert_sparse_cross_attention_mask_to_dense(
            cross_attention_token_mask,
            num_tiles,
            self.image_processor.max_image_tiles,
            encoding.len(),
        )?;

        let encoding_len = encoding.len();
        let encoding = Tensor::from_vec(encoding, (1, encoding_len), &Device::Cpu)?;

        let output = MllamaProcessorOutput {
            input_ids: encoding.clone(),
            attention_mask: Tensor::ones_like(&encoding)?,
            pixel_values: image,
            aspect_ratio_ids: aspect_ratio_ids,
            aspect_ratio_mask: aspect_ratio_mask,
            cross_attention_mask: cross_attention_mask,
        };

        Ok(output)
    }
}
