// #[path = "config.rs"]
// mod config;

// use candle_nn::Conv2d;
use crate::config::MllamaVisionConfig;
use candle::{DType, IndexOp, Result, Tensor, D};
use candle_nn::{embedding, Embedding, LayerNorm, Linear, Module, VarBuilder};
use imageproc::{binary_descriptors::brief::TestPair, definitions::Image};
use serde::{Deserialize, Serialize};

fn scaled_dot_product_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dim = q.dim(D::Minus1)?;
    let scale_factor = 1.0 / (dim as f64).sqrt();
    let attn_weights = (q.matmul(&k.t()?)? * scale_factor)?;
    candle_nn::ops::softmax_last_dim(&attn_weights)?.matmul(v)
}

pub struct MllamaVisionModel {
    pub image_size: usize,
    pub patch_size: usize,
    pub max_num_tiles: usize,
    pub hidden_size: usize,
    pub num_channels: usize,
    pub intermediate_layers_indices: Vec<i32>,
    pub num_patches: usize,
    pub scale: f32,
    pub patch_embedding: candle_nn::Conv2d,
    pub class_embedding: Tensor,
    pub gated_positional_embedding: MllamaPrecomputedPositionEmbedding,
    pub pre_tile_positional_embedding: MllamaPrecomputedAspectRatioEmbedding,
    pub post_tile_positional_embedding: MllamaPrecomputedAspectRatioEmbedding,
    pub layernorm_pre: LayerNorm,
    pub layernorm_post: LayerNorm,
    pub transformer: MllamaVisionEncoder,
    pub global_transformer: MllamaVisionEncoder,
}
impl MllamaVisionModel {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig) -> Result<Self> {
        let image_size = cfg.image_size;
        let patch_size = cfg.patch_size;
        let max_num_tiles = cfg.max_num_tiles;
        let hidden_size = cfg.hidden_size;
        let num_channels = cfg.num_channels;
        let intermediate_layers_indices = cfg.intermediate_layers_indices.clone();
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
        let scale = (cfg.hidden_size as f32).powf(-0.5);
        let patch_embedding = candle_nn::conv2d(
            cfg.num_channels,
            hidden_size,
            patch_size,
            candle_nn::Conv2dConfig {
                padding: 0,
                stride: patch_size,
                dilation: 1 as usize,
                groups: 1 as usize,
            },
            vb.pp("patch_embedding"),
        )?;
        let class_embedding = vb.get(hidden_size, "class_embedding")?;
        let gated_positional_embedding =
            MllamaPrecomputedPositionEmbedding::new(vb.pp("gated_positional_embedding"), cfg)?;

        let pre_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding::new(
            vb.pp("pre_tile_positional_embedding"),
            cfg,
            Some(true),
        )?;
        let post_tile_positional_embedding = MllamaPrecomputedAspectRatioEmbedding::new(
            vb.pp("post_tile_positional_embedding"),
            cfg,
            Some(true),
        )?;

        // layer norms
        let layernorm_pre = candle_nn::layer_norm(
            hidden_size,
            candle_nn::LayerNormConfig::default(),
            vb.pp("layernorm_pre"),
        )?;
        let layernorm_post = candle_nn::layer_norm(
            hidden_size,
            candle_nn::LayerNormConfig::default(),
            vb.pp("layernorm_post"),
        )?;

        // encoders
        let transformer = MllamaVisionEncoder::new(
            vb.pp("transformer"),
            cfg,
            Some(cfg.num_hidden_layers),
            Some(false),
        )?;
        let global_transformer = MllamaVisionEncoder::new(
            vb.pp("global_transformer"),
            cfg,
            Some(cfg.num_hidden_layers),
            Some(false),
        )?;

        Ok(Self {
            image_size,
            patch_size,
            max_num_tiles,
            hidden_size,
            num_channels,
            intermediate_layers_indices,
            num_patches,
            scale,
            patch_embedding,
            class_embedding,
            gated_positional_embedding,
            pre_tile_positional_embedding,
            post_tile_positional_embedding,
            layernorm_pre,
            layernorm_post,
            transformer,
            global_transformer,
        })
    }

    pub fn apply_class_embedding(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let [batch_size, _, hidden_size] = *hidden_state.dims() else {
            panic!("wrong dim")
        };
        let class_embedding = self.class_embedding.expand((batch_size, 1, hidden_size))?;
        let hidden_state = Tensor::cat(&[class_embedding, hidden_state.clone()], 1)?;

        Ok(hidden_state)
    }

    pub fn prepare_aspect_ratio_attention_mask(
        &self,
        aspect_ratio_mask: &Tensor,
        num_patches: usize,
        target_length: usize,
    ) -> Result<Tensor> {
        // Expand aspect ratio mask to target_length
        let [batch_size, max_num_tiles] = *aspect_ratio_mask.dims() else {
            panic!("wrong dim")
        };
        let attention_mask = aspect_ratio_mask.reshape((batch_size, max_num_tiles, 1, 1))?;
        let attention_mask = attention_mask.repeat((1, 1, target_length, 1))?;

        // Mask padding patches
        let pad_patches = target_length - num_patches;
        // attention_mask[:, :, -pad_patches:] = 0
        let attention_mask = attention_mask.slice_assign(
            &[
                0..attention_mask.dims()[0],
                0..attention_mask.dims()[1],
                attention_mask.dims()[2] - pad_patches..attention_mask.dims()[2],
            ],
            &attention_mask.zeros_like()?,
        )?;

        // Invert the mask (0 -> 1, 1 -> 0)
        let attention_mask = (attention_mask.ones_like()? - attention_mask)?;

        // Reshape to 2D and create 4D attention mask
        // (batch_size, 1, max_num_tiles * target_length, max_num_tiles * target_length)
        let attention_mask =
            attention_mask.reshape((batch_size, max_num_tiles * target_length, 1))?;
        let attention_mask = attention_mask.matmul(&attention_mask.transpose(
            attention_mask.dims()[attention_mask.dims().len() - 1],
            attention_mask.dims()[attention_mask.dims().len() - 2],
        )?)?;
        let attention_mask = attention_mask.unsqueeze(1)?;

        Ok(attention_mask)
    }
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        aspect_ratio_ids: &Tensor,
        aspect_ratio_mask: &Tensor,
    ) -> Result<Tensor> {
        let [batch_size, num_concurrent_media, num_tiles, num_channels, height, width] =
            *pixel_values.dims()
        else {
            panic!("wrong length")
        };
        let pixel_values = pixel_values.reshape((
            batch_size * num_concurrent_media * num_tiles,
            num_channels,
            height,
            width,
        ))?;
        let aspect_ratio_ids = aspect_ratio_ids.reshape((
            batch_size * num_concurrent_media,
            aspect_ratio_ids.elem_count() / (batch_size * num_concurrent_media),
        ))?;

        // Patch embedding
        let patch_embeds = self.patch_embedding.forward(&pixel_values)?;
        // let hidden_state = patch_embeds.flatten(2).transpose(1, 2)
        let hidden_state = patch_embeds.flatten(2, 2)?.transpose(1, 2)?;

        // Tile embeddings
        let [_, num_patches, dim] = *hidden_state.dims() else {
            panic!("wrong shape")
        };
        let hidden_state = hidden_state.reshape((
            batch_size * num_concurrent_media,
            num_tiles,
            hidden_state.elem_count() / (batch_size * num_concurrent_media * num_tiles * dim),
            dim,
        ))?;
        let hidden_state = self
            .pre_tile_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids)?;

        // Add cls token
        let hidden_state = hidden_state.reshape((
            batch_size * num_concurrent_media * num_tiles,
            num_patches,
            dim,
        ))?;
        let hidden_state = self.apply_class_embedding(&hidden_state)?;
        let mut num_patches = num_patches + 1;

        // Position embeddings
        let hidden_state = hidden_state.reshape((
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches,
            dim,
        ))?;
        let hidden_state = self
            .gated_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids)?;

        let hidden_state = self.layernorm_pre.forward(&hidden_state)?;

        // Compute the number of tokens to pad
        let num_padding_patches =
            (8 - (hidden_state.dims()[hidden_state.dims().len() - 2] % 8)) % 8;
        // Compute padding tuple for pad function
        // padding = (0, 0, 0, num_padding_patches)  // (pad_left, pad_right, pad_left for dim -2, pad_right for dim -2)
        // Pad the tensor
        hidden_state.pad_with_zeros(
            hidden_state.dims()[hidden_state.dims().len() - 2],
            0,
            num_padding_patches,
        )?;

        let slice_index: Option<i32> = if num_padding_patches > 0 {
            Some(num_padding_patches as i32 * -1)
        } else {
            None
        };

        // Prepare attention mask
        let attention_mask = aspect_ratio_mask.reshape((
            batch_size * num_concurrent_media,
            aspect_ratio_mask.elem_count() / (batch_size * num_concurrent_media),
        ))?;
        let attention_mask = self.prepare_aspect_ratio_attention_mask(
            &attention_mask,
            self.num_patches,
            hidden_state.dims()[2],
        )?;

        // Apply encoder
        let hidden_state = hidden_state.reshape((
            batch_size * num_concurrent_media,
            hidden_state.elem_count() / (batch_size * num_concurrent_media * dim),
            dim,
        ))?;
        let (hidden_state, transformer_encoder_states) =
            self.transformer.forward(&hidden_state, Some(true))?;

        let hidden_state = self.layernorm_post.forward(&hidden_state)?;

        // Apply global encoder
        let hidden_state = hidden_state.reshape((
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        ))?;
        let hidden_state = self
            .post_tile_positional_embedding
            .forward(&hidden_state, &aspect_ratio_ids)?;
        let hidden_state = hidden_state.reshape((
            batch_size * num_concurrent_media,
            num_tiles * (num_patches + num_padding_patches),
            dim,
        ))?;
        let (hidden_state, _) = self.global_transformer.forward(&hidden_state, None)?;

        // Remove padding form hidden state
        let hidden_state = hidden_state.reshape((
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            dim,
        ))?;
        // Tensor::gather(&self, indexes, dim)
        let hidden_state = match slice_index {
            Some(idx) => hidden_state.i((.., .., ..(idx as usize)))?,
            None => hidden_state,
        };
        let hidden_state = hidden_state.reshape((
            batch_size,
            num_concurrent_media,
            num_tiles,
            num_patches,
            dim,
        ))?;

        // Collect intermediate layer outputs from encoder output
        let mut all_intermediate_hidden_states = Vec::new();
        for i in self.intermediate_layers_indices.iter() {
            all_intermediate_hidden_states.push(transformer_encoder_states[*i as usize].clone());
        }
        let intermediate_hidden_states = Tensor::stack(
            &all_intermediate_hidden_states,
            all_intermediate_hidden_states[0].dims().len(),
        )?;

        // Remove padding from intermediate hidden states
        let intermediate_hidden_states = intermediate_hidden_states.reshape((
            batch_size * num_concurrent_media,
            num_tiles,
            num_patches + num_padding_patches,
            intermediate_hidden_states.elem_count() / (batch_size * num_concurrent_media)
                * (num_tiles)
                * (num_patches + num_padding_patches),
        ))?;
        let hidden_state = match slice_index {
            Some(idx) => hidden_state.i((.., .., ..(idx as usize)))?,
            None => hidden_state,
        };
        let intermediate_hidden_states = match slice_index {
            Some(idx) => intermediate_hidden_states.i((.., .., ..(idx as usize)))?,
            None => intermediate_hidden_states,
        };
        let intermediate_hidden_states = intermediate_hidden_states.reshape((
            batch_size,
            num_concurrent_media,
            num_tiles,
            num_patches,
            intermediate_hidden_states.elem_count()
                / (batch_size * num_concurrent_media * num_tiles * num_patches),
        ))?;

        // Concatenate final hidden state and intermediate hidden states
        let hidden_state = Tensor::cat(
            &[&hidden_state, &intermediate_hidden_states],
            hidden_state.dims().len() - 1,
        )?;

        Ok(hidden_state)
    }
}

pub struct MllamaPrecomputedAspectRatioEmbedding {
    pub max_num_tiles: usize,
    pub hidden_size: usize,
    pub max_aspect_ratio_id: usize,
    pub is_gated: bool,
    pub embedding: candle_nn::Embedding,
    pub gate: Tensor,
}
impl MllamaPrecomputedAspectRatioEmbedding {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig, is_gated: Option<bool>) -> Result<Self> {
        let max_num_tiles = cfg.max_num_tiles;
        let hidden_size = cfg.hidden_size;
        let max_aspect_ratio_id = cfg.supported_aspect_ratios.len();
        let is_gated = if is_gated.is_some() {
            is_gated.unwrap()
        } else {
            true
        };
        let embedding = candle_nn::embedding(
            max_aspect_ratio_id + 1,
            max_num_tiles * hidden_size,
            vb.pp("embedding"),
        )?;
        let gate = vb.get(1, "gate")?;

        Ok(Self {
            max_num_tiles,
            hidden_size,
            max_aspect_ratio_id,
            is_gated,
            embedding,
            gate,
        })
    }

    pub fn forward(&self, hidden_state: &Tensor, aspect_ratio_ids: &Tensor) -> Result<Tensor> {
        let mut embedding = self.embedding.forward(aspect_ratio_ids)?;
        embedding = embedding.reshape((
            embedding.elem_count() / self.max_num_tiles * self.hidden_size,
            self.max_num_tiles,
            1,
            self.hidden_size,
        ))?;

        if self.is_gated == true {
            embedding = embedding.broadcast_mul(&self.gate.tanh()?)?;
        }

        let hidden_state = (hidden_state + embedding)?;
        Ok(hidden_state)
    }
}

pub struct MllamaPrecomputedPositionEmbedding {
    max_num_tiles: usize,
    max_aspect_ratio_id: usize,
    num_patches: usize,
    hidden_size: usize,
    scale: Tensor,
    gate: Tensor,
    embedding: Tensor,
    tile_embedding: Embedding,
}
impl MllamaPrecomputedPositionEmbedding {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig) -> Result<Self> {
        let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
        let scale = Tensor::new((cfg.hidden_size as f32).powf(-0.5), vb.device())?;

        let gate = vb.get(1, "gate")?;
        // do we need to have: self.embedding = nn.Parameter(self.scale * position_embedding)????
        // I think we don't need to multiply possition_embedding here because
        // after training self.embedding already has the proper value(we already did multiply it with the scale varibale)
        let embedding = vb.get((num_patches, cfg.hidden_size), "embedding")?;
        let tile_embedding = candle_nn::embedding(
            cfg.supported_aspect_ratios.len() + 1,
            cfg.max_num_tiles * num_patches * cfg.hidden_size,
            vb.pp("tile_embedding"),
        )?;

        Ok(Self {
            max_num_tiles: cfg.max_num_tiles,
            max_aspect_ratio_id: cfg.supported_aspect_ratios.len(),
            num_patches: num_patches,
            hidden_size: cfg.hidden_size,
            scale: scale,
            gate: gate,
            embedding: embedding,
            tile_embedding: tile_embedding,
        })
    }

    pub fn forward(&self, hidden_state: &Tensor, aspect_ratio_ids: &Tensor) -> Result<Tensor> {
        // position embeddings
        let gated_position_embedding = ((Tensor::ones(1, self.gate.dtype(), self.gate.device())
            - self.gate.tanh()?)
            * &self.embedding)?;

        let mut hidden_state = (hidden_state
            + gated_position_embedding.reshape((1, 1, self.num_patches, self.hidden_size)))?;

        // precomputed tile position embeddings
        let mut tile_posistion_embedding = self.tile_embedding.forward(aspect_ratio_ids)?;
        let batch_size = hidden_state.dim(0)?;
        tile_posistion_embedding = tile_posistion_embedding.reshape((
            batch_size,
            self.max_num_tiles,
            self.num_patches,
            self.hidden_size,
        ))?;
        let gated_tile_position_embedding =
            (self.gate.tanh()?).broadcast_mul(&tile_posistion_embedding)?;
        hidden_state = (hidden_state + gated_tile_position_embedding)?;
        Ok(hidden_state)
    }
}

pub struct MllamaVisionSdpaAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
}
impl MllamaVisionSdpaAttention {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.hidden_size;
        let q_proj = Linear::new(
            vb.get((embed_dim, num_heads * head_dim), "q_proj.weight")?,
            None,
        );
        let k_proj = Linear::new(
            vb.get((embed_dim, num_heads * head_dim), "k_proj.weight")?,
            None,
        );
        let v_proj = Linear::new(
            vb.get((embed_dim, num_heads * head_dim), "v_proj.weight")?,
            None,
        );
        let o_proj = Linear::new(
            vb.get((embed_dim, num_heads * head_dim), "o_proj.weight")?,
            None,
        );
        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
        })
    }

    pub fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        let mut query = self.q_proj.forward(&hidden_state)?;
        let mut key = self.k_proj.forward(&hidden_state)?;
        let mut value = self.v_proj.forward(&hidden_state)?;

        let batch_size = query.dim(0)?;
        let q_seq_len = query.dim(1)?;
        let kv_seq_len = key.dim(1)?;

        query = query.reshape((batch_size, q_seq_len, self.num_heads, self.head_dim))?;
        key = key.reshape((batch_size, kv_seq_len, self.num_heads, self.head_dim))?;
        value = value.reshape((batch_size, kv_seq_len, self.num_heads, self.head_dim))?;

        query = query.transpose(1, 2)?;
        key = key.transpose(1, 2)?;
        value = value.transpose(1, 2)?;

        let mut attn_output = scaled_dot_product_attention(&query, &key, &value)?;
        attn_output = attn_output.transpose(1, 2)?;

        attn_output = attn_output.reshape((
            batch_size,
            q_seq_len,
            attn_output.elem_count() / (batch_size * q_seq_len),
        ))?;

        let output = self.o_proj.forward(&attn_output)?;
        Ok(output)
    }
}

pub struct MllamaVisionMLP {
    fc1: Linear,
    fc2: Linear,
}
impl MllamaVisionMLP {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig) -> Result<Self> {
        let fc1 = Linear::new(
            vb.get((cfg.hidden_size, cfg.intermediate_size), "fc1.weight")?,
            Some(vb.get(cfg.hidden_size, "fc1.bias")?),
        );
        let fc2 = Linear::new(
            vb.get((cfg.intermediate_size, cfg.hidden_size), "fc2.weight")?,
            Some(vb.get(cfg.intermediate_size, "fc2.bias")?),
        );
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.fc1.forward(hidden_states)?;
        hidden_states = hidden_states.gelu()?;
        hidden_states = self.fc2.forward(&hidden_states)?;
        Ok(hidden_states)
    }
}

pub struct MllamaVisionEncoderLayer {
    hidden_size: usize,
    num_attention_heads: usize,
    is_gated: bool,
    intermediate_size: usize,

    self_attn: MllamaVisionSdpaAttention,
    mlp: MllamaVisionMLP,

    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,

    gate_attn: Option<Tensor>,
    gate_ffn: Option<Tensor>,
}
impl MllamaVisionEncoderLayer {
    pub fn new(vb: VarBuilder, cfg: &MllamaVisionConfig, is_gated: Option<bool>) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let num_attention_heads = cfg.num_attention_heads;
        let intermediate_size = cfg.intermediate_size;

        let self_attn = MllamaVisionSdpaAttention::new(vb.pp("self_attn"), cfg)?;
        let mlp = MllamaVisionMLP::new(vb.pp("mlp"), cfg)?;

        let input_layernorm = LayerNorm::new(
            vb.get(hidden_size, "input_layernorm.weight")?,
            vb.get(hidden_size, "input_layernorm.bias")?,
            cfg.norm_eps as f64,
        );
        let post_attention_layernorm = LayerNorm::new(
            vb.get(hidden_size, "post_attention_layernorm.weight")?,
            vb.get(hidden_size, "post_attention_layernorm.bias")?,
            cfg.norm_eps as f64,
        );

        let (is_gated, gate_attn, gate_ffn) = if let Some(true) = is_gated {
            let gate_attn = vb.get(1, "gate_attn")?;
            let gate_ffn = vb.get(1, "gate_ffn")?;
            (true, Some(gate_attn), Some(gate_ffn))
        } else {
            (false, None, None)
        };

        Ok(Self {
            hidden_size,
            num_attention_heads,
            is_gated,
            intermediate_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            gate_attn,
            gate_ffn,
        })
    }

    pub fn forward(&self, hidden_state: &Tensor) -> Result<Tensor> {
        // Self Attention
        let residual = hidden_state.clone();
        let mut hidden_state = self.input_layernorm.forward(&hidden_state)?;
        hidden_state = self.self_attn.forward(&hidden_state)?;

        hidden_state = if self.is_gated {
            self.gate_attn
                .as_ref()
                .unwrap()
                .tanh()?
                .broadcast_mul(&hidden_state)?
        } else {
            hidden_state
        };
        hidden_state = (residual + hidden_state)?;

        // Feed Forward
        let residual = hidden_state.clone();
        hidden_state = self.post_attention_layernorm.forward(&hidden_state)?;
        hidden_state = self.mlp.forward(&hidden_state)?;
        hidden_state = if self.is_gated {
            self.gate_ffn
                .as_ref()
                .unwrap()
                .tanh()?
                .broadcast_mul(&hidden_state)?
        } else {
            hidden_state
        };
        hidden_state = (residual + hidden_state)?;
        Ok(hidden_state)
    }
}

pub struct MllamaVisionEncoder {
    pub layers: Vec<MllamaVisionEncoderLayer>,
}
impl MllamaVisionEncoder {
    pub fn new(
        vb: VarBuilder,
        cfg: &MllamaVisionConfig,
        num_layers: Option<usize>,
        is_gated: Option<bool>,
    ) -> Result<Self> {
        let num_layers = num_layers.unwrap_or(32);
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer_name = format!("layers.{}", i);
            layers.push(MllamaVisionEncoderLayer::new(
                vb.pp(layer_name),
                cfg,
                is_gated,
            )?);
        }
        Ok(Self { layers })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        output_hidden_states: Option<bool>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let mut hidden_states = hidden_states.clone();
        let mut encoder_states = Vec::new();
        for layer in &self.layers {
            if let Some(true) = output_hidden_states {
                encoder_states.push(hidden_states.clone());
            }
            hidden_states = layer.forward(&hidden_states)?;
        }
        if let Some(true) = output_hidden_states {
            encoder_states.push(hidden_states.clone());
        }
        Ok((hidden_states, encoder_states))
    }
}
