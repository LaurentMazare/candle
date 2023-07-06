use super::*;
use candle::{safetensors::SafeTensors, Device, Result, Tensor};
use std::path::PathBuf;

pub struct VarBuilder<'a> {
    routing: HashMap<String, usize>,
    safetensors: Vec<SafeTensors<'a>>,
    device: Device,
}

impl<'a> VarBuilder<'a> {
    pub fn new(safetensors: Vec<SafeTensors<'a>>, device: Device) -> Self {
        let mut routing = HashMap::new();
        for (index, sf) in safetensors.iter().enumerate() {
            for k in sf.names() {
                routing.insert(k.to_string(), index);
            }
        }

        Self {
            safetensors,
            device,
            routing,
        }
    }

    pub fn get(&self, tensor_name: &str) -> Result<Tensor> {
        // Unwrap or 0  just to let the proper error flow.
        let index = self.routing.get(tensor_name).unwrap_or(&0);
        self.safetensors[*index]
            .tensor(tensor_name, &self.device)?
            .to_dtype(DTYPE)
    }
}

impl Linear {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let weight = vb.get(&format!("{prefix}.weight"))?;
        Ok(Self::new(weight))
    }

    fn load_multi(prefixes: &[&str], vb: &VarBuilder) -> Result<Self> {
        let weights: Vec<_> = prefixes
            .iter()
            .map(|p| vb.get(&format!("{p}.weight")).unwrap())
            .collect();
        let weight = Tensor::cat(&weights, 0)?;
        Ok(Self::new(weight))
    }
}

impl RmsNorm {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let scale = vb.get(&format!("{prefix}.weight"))?;
        Ok(Self::new(scale))
    }
}

impl CausalSelfAttention {
    fn load(prefix: &str, vb: &VarBuilder, cache: &Cache, config: &Config) -> Result<Self> {
        let c_attn = Linear::load_multi(
            &[
                &format!("{prefix}.q_proj"),
                &format!("{prefix}.k_proj"),
                &format!("{prefix}.v_proj"),
            ],
            vb,
        )?;
        let o_proj = Linear::load(&format!("{prefix}.o_proj"), vb)?;
        Ok(Self::new(c_attn, o_proj, config.n_head, cache))
    }
}

impl Mlp {
    fn load(prefix: &str, vb: &VarBuilder) -> Result<Self> {
        let c_fc1 = Linear::load(&format!("{prefix}.gate_proj"), vb)?;
        let c_fc2 = Linear::load(&format!("{prefix}.up_proj"), vb)?;
        let c_proj = Linear::load(&format!("{prefix}.down_proj"), vb)?;
        Ok(Self::new(c_fc1, c_fc2, c_proj))
    }
}

impl Block {
    fn load(prefix: &str, vb: &VarBuilder, cache: &Cache, config: &Config) -> Result<Self> {
        let attn = CausalSelfAttention::load(&format!("{prefix}.self_attn"), vb, cache, config)?;
        let mlp = Mlp::load(&format!("{prefix}.mlp"), vb)?;
        let input_layernorm = RmsNorm::load(&format!("{prefix}.input_layernorm"), vb)?;
        let post_attention_layernorm =
            RmsNorm::load(&format!("{prefix}.post_attention_layernorm"), vb)?;
        Ok(Self::new(
            input_layernorm,
            attn,
            post_attention_layernorm,
            mlp,
        ))
    }
}

impl Llama {
    pub fn load(
        device: &Device,
        filenames: &[PathBuf],
        cache: &Cache,
        config: &Config,
    ) -> Result<Self> {
        let handles: Vec<_> = filenames
            .iter()
            .map(|f| unsafe { candle::safetensors::MmapedFile::new(f) })
            .collect::<Result<Vec<_>>>()?;
        let tensors: Vec<_> = handles
            .iter()
            .map(|h| h.deserialize())
            .collect::<Result<Vec<_>>>()?;

        let vb = VarBuilder::new(tensors, device.clone());

        let embedding = vb.get("model.embed_tokens.weight")?;
        let wte = Embedding::new(embedding);
        let lm_head = Linear::load("lm_head", &vb)?;
        let norm = RmsNorm::load("model.norm", &vb)?;
        let blocks: Vec<_> = (0..config.n_layer)
            .map(|i| Block::load(&format!("model.layers.{i}"), &vb, cache, config).unwrap())
            .collect();

        Ok(Self::new(wte, blocks, norm, lm_head))
    }
}
