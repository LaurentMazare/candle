use candle::{Result, Tensor};

pub struct MllamaForCausalLM {
    // first we need MllamaTextModel
}

pub struct KVCache {}

pub trait AttentionDecoderLayer {
    fn forward(
        hidden_states: &Tensor,
        cross_attention_states: Option<Tensor>,
        cross_attention_mask: Option<Tensor>,
        attention_mask: Option<Tensor>,
        full_text_row_masked_out_mask: Option<(Tensor, Tensor)>,
        position_ids: Option<Tensor>,
        past_key_value: Option<Cache>,
        output_attentions: Option<bool>,
        use_cache: Option<bool>,
        cache_position: Option<Tensor>,
        position_embeddings: Option<(Tensor, Tensor)>,
    ) -> Result<Vec<Tensor>>;
}

pub struct MllamaTextModel {}
impl MllamaTextModel {}
