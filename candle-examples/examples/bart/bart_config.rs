use std::collections::HashMap;

use serde::Deserialize;

use candle_nn::Activation;

#[derive(Debug, Deserialize, Clone)]
/// # BART model configuration
/// Defines the BART model architecture (e.g. number of layers, hidden layer size, label mapping...)
pub struct Config {
    pub num_labels: Option<usize>,
    pub activation_function: Option<Activation>,
    pub activation_dropout: f32,
    pub attention_dropout: f32,
    pub classif_dropout: Option<f32>,
    pub d_model: usize,
    pub decoder_attention_heads: usize,
    pub decoder_ffn_dim: usize,
    pub decoder_layerdrop: f32,
    pub decoder_layers: usize,
    pub decoder_start_token_id: Option<usize>,
    pub dropout: f32,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub encoder_layerdrop: f32,
    pub encoder_layers: usize,
    pub bos_token_id: Option<usize>,
    pub eos_token_id: Option<usize>,
    pub forced_bos_token_id: Option<usize>,
    pub forced_eos_token_id: Option<usize>,
    pub pad_token_id: Option<usize>,
    pub id2label: Option<HashMap<usize, String>>,
    pub label2id: Option<HashMap<String, usize>>,
    pub init_std: f32,
    pub is_decoder: Option<bool>,
    pub is_encoder_decoder: Option<bool>,
    pub max_position_embeddings: usize,
    pub min_length: Option<usize>,
    pub no_repeat_ngram_size: Option<usize>,
    pub normalize_embedding: Option<bool>,
    pub num_hidden_layers: usize,
    pub output_attentions: Option<bool>,
    pub output_hidden_states: Option<bool>,
    pub output_past: Option<bool>,
    pub static_position_embeddings: Option<bool>,
    pub scale_embedding: Option<bool>,
    pub vocab_size: usize,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            num_labels: Some(3),
            activation_function: Some(Activation::Gelu),
            activation_dropout: 0.0,
            attention_dropout: 0.0,
            classif_dropout: Some(0.0),
            d_model: 1024,
            decoder_attention_heads: 16,
            decoder_ffn_dim: 4096,
            decoder_layerdrop: 0.0,
            decoder_layers: 12,
            decoder_start_token_id: Some(2),
            dropout: 0.1,
            encoder_attention_heads: 16,
            encoder_ffn_dim: 4096,
            encoder_layerdrop: 0.0,
            encoder_layers: 12,
            bos_token_id: Some(0),
            eos_token_id: Some(2),
            pad_token_id: Some(1),
            forced_bos_token_id: Some(0),
            forced_eos_token_id: Some(2),
            id2label: None,
            label2id: None,
            init_std: 0.02,
            is_decoder: None,
            is_encoder_decoder: Some(true),
            max_position_embeddings: 1024,
            min_length: None,
            no_repeat_ngram_size: None,
            normalize_embedding: Some(true),
            num_hidden_layers: 12,
            output_attentions: None,
            output_hidden_states: None,
            output_past: None,
            static_position_embeddings: None,
            scale_embedding: Some(false),
            vocab_size: 50265,
        }
    }
}
