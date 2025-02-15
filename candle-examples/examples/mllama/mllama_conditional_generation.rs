use candle::{DType, IndexOp, Module, Result, Tensor, D};
use candle_nn::{linear, Linear, VarBuilder};

use crate::config::MllamaConfig;
use crate::vision_model::MllamaVisionModel;

pub struct MllamaForConditionalGeneration {
    vocab_size: usize,
    hidden_size: usize,
    max_num_tiles: usize,
    vision_output_dim: usize,
    // pad_token_id: i32,
    vision_model: MllamaVisionModel,
    multi_modal_projector: Linear,
}
impl MllamaForConditionalGeneration {
    pub fn new(vb: VarBuilder, cfg: &MllamaConfig) -> Result<Self> {
        let vocab_size = cfg.text_config.vocab_size;
        let hidden_size = cfg.text_config.hidden_size;
        let max_num_tiles = cfg.vision_config.max_num_tiles;
        let vision_output_dim = cfg.vision_config.vision_output_dim;
        // let pad_token_id = match cfg self.config.pad_token_id if self.config.pad_token_id is not None else -1

        let vision_model = MllamaVisionModel::new(vb.pp("vision_model"), &cfg.vision_config)?;

        let multi_modal_projector = linear(
            cfg.vision_config.vision_output_dim,
            cfg.text_config.hidden_size,
            vb.pp("multi_modal_projector"),
        )?;
        Ok(Self {
            vocab_size,
            hidden_size,
            max_num_tiles,
            vision_output_dim,
            vision_model,
            multi_modal_projector,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        aspect_ratio_ids: &Tensor,
        aspect_ratio_mask: &Tensor,
    ) -> Result<()> {
        let vision_outputs =
            self.vision_model
                .forward(pixel_values, aspect_ratio_ids, aspect_ratio_mask)?;

        let cross_attention_states = self.multi_modal_projector.forward(&vision_outputs)?;
        let cross_attention_states = cross_attention_states.reshape((
            cross_attention_states.elem_count()
                / (cross_attention_states.dims()[cross_attention_states.dims().len() - 2]
                    * self.hidden_size),
            cross_attention_states.dims()[cross_attention_states.dims().len() - 2],
            self.hidden_size,
        ))?;

        todo!()
    }
}
