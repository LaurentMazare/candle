use candle::op::Op;
use candle::quantized::{gguf_file, GgmlDType, QTensor};
use candle::{Device, Result, Tensor};
use clap::{Parser, Subcommand, ValueEnum};
use rayon::prelude::*;
use safetensors::tensor;
use serde_json;

#[derive(ValueEnum, Debug, Clone)]
enum QuantizationMode {
    /// The default quantization includes all 2d tensors, except the output tensor which always
    /// uses Q6_K.
    Llama,
}

impl QuantizationMode {
    fn quantize(
        &self,
        name: &str,
        tensor: QTensor,
        dtype: GgmlDType,
        bitnet_mode: bool,
    ) -> Result<QTensor> {
        match self {
            Self::Llama => {
                // Same behavior as the llama.cpp quantization.
                let should_quantize = name.ends_with(".weight") && tensor.rank() == 2;
                if should_quantize {
                    let tensor = tensor.dequantize(&Device::Cpu)?;
                    if name == "output.weight" {
                        QTensor::quantize(&tensor, GgmlDType::Q6K)
                    } else {
                        QTensor::quantize(&tensor, dtype)
                    }
                } else {
                    Ok(tensor)
                }
            }
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
enum Quantization {
    #[value(name = "q4_0")]
    Q4_0,
    #[value(name = "q4_1")]
    Q4_1,
    #[value(name = "q5_0")]
    Q5_0,
    #[value(name = "q5_1")]
    Q5_1,
    #[value(name = "q8_0")]
    Q8_0,
    #[value(name = "q8_1")]
    Q8_1,
    #[value(name = "q2b0")]
    Q2b0,
    #[value(name = "q2b1")]
    Q2b1,
    #[value(name = "qi8")]
    QI8,
    Q2k,
    Q3k,
    Q4k,
    Q5k,
    Q6k,
    Q8k,
    F16,
    F32,
}

impl Quantization {
    fn dtype(&self) -> GgmlDType {
        match self {
            Quantization::Q4_0 => GgmlDType::Q4_0,
            Quantization::Q4_1 => GgmlDType::Q4_1,
            Quantization::Q5_0 => GgmlDType::Q5_0,
            Quantization::Q5_1 => GgmlDType::Q5_1,
            Quantization::Q8_0 => GgmlDType::Q8_0,
            Quantization::Q8_1 => GgmlDType::Q8_1,
            Quantization::Q2k => GgmlDType::Q2K,
            Quantization::Q3k => GgmlDType::Q3K,
            Quantization::Q4k => GgmlDType::Q4K,
            Quantization::Q5k => GgmlDType::Q5K,
            Quantization::Q6k => GgmlDType::Q6K,
            Quantization::Q8k => GgmlDType::Q8K,
            Quantization::F16 => GgmlDType::F16,
            Quantization::F32 => GgmlDType::F32,
            Quantization::Q2b0 => GgmlDType::Q2b0,
            Quantization::QI8 => GgmlDType::QI8,
            Quantization::Q2b1 => GgmlDType::Q2b1,
        }
    }
}

#[derive(ValueEnum, Debug, Clone)]
enum Format {
    Safetensors,
    Npz,
    Ggml,
    Gguf,
    Pth,
    Pickle,
}

impl Format {
    fn infer<P: AsRef<std::path::Path>>(p: P) -> Option<Self> {
        p.as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .and_then(|e| match e {
                // We don't infer any format for .bin as it can be used for ggml/gguf or pytorch.
                "safetensors" | "safetensor" => Some(Self::Safetensors),
                "npz" => Some(Self::Npz),
                "pth" | "pt" => Some(Self::Pth),
                "ggml" => Some(Self::Ggml),
                "gguf" => Some(Self::Gguf),
                _ => None,
            })
    }
}

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Ls {
        files: Vec<std::path::PathBuf>,

        /// The file format to use, if unspecified infer from the file extension.
        #[arg(long, value_enum)]
        format: Option<Format>,

        /// Enable verbose mode.
        #[arg(short, long)]
        verbose: bool,
    },

    Print {
        file: std::path::PathBuf,

        names: Vec<String>,

        /// The file format to use, if unspecified infer from the file extension.
        #[arg(long, value_enum)]
        format: Option<Format>,

        /// Print the whole content of each tensor.
        #[arg(long)]
        full: bool,

        /// Line width for printing the tensors.
        #[arg(long)]
        line_width: Option<usize>,
    },

    Quantize {
        /// The input file(s), in safetensors format.
        in_file: Vec<std::path::PathBuf>,

        /// The output file, in gguf format.
        #[arg(long)]
        out_file: std::path::PathBuf,

        #[clap(long, short, action)]
        bitnet_mode: bool,

        // Allow to specify quantization_bitnet in case of bitnet_mode
        #[arg(long, value_enum)]
        bitnet_quantization: Option<Quantization>,

        /// The quantization schema to apply.
        #[arg(long, value_enum)]
        quantization: Quantization,

        /// Which tensor to quantize.
        #[arg(long, value_enum, default_value_t = QuantizationMode::Llama)]
        mode: QuantizationMode,
    },

    Dequantize {
        /// The input file, in gguf format.
        in_file: std::path::PathBuf,

        /// The output file, in safetensors format.
        #[arg(long)]
        out_file: std::path::PathBuf,
    },
}

#[derive(Parser, Debug, Clone)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

fn run_print(
    file: &std::path::PathBuf,
    names: Vec<String>,
    format: Option<Format>,
    full: bool,
    line_width: Option<usize>,
    device: &Device,
) -> Result<()> {
    if full {
        candle::display::set_print_options_full();
    }
    if let Some(line_width) = line_width {
        candle::display::set_line_width(line_width)
    }
    let format = match format {
        Some(format) => format,
        None => match Format::infer(file) {
            Some(format) => format,
            None => {
                println!(
                    "{file:?}: cannot infer format from file extension, use the --format flag"
                );
                return Ok(());
            }
        },
    };
    match format {
        Format::Npz => {
            let tensors = candle::npy::NpzTensors::new(file)?;
            let names = if names.is_empty() {
                tensors.names().into_iter().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match tensors.get(name)? {
                    Some(tensor) => println!("{tensor}"),
                    None => println!("not found"),
                }
            }
        }
        Format::Safetensors => {
            use candle::safetensors::Load;
            let tensors = unsafe { candle::safetensors::MmapedSafetensors::new(file)? };
            let tensors: std::collections::HashMap<_, _> = tensors.tensors().into_iter().collect();
            let names = if names.is_empty() {
                tensors.keys().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match tensors.get(name) {
                    Some(tensor_view) => {
                        let tensor = tensor_view.load(device)?;
                        println!("{tensor}")
                    }
                    None => println!("not found"),
                }
            }
        }
        Format::Pth => {
            let pth_file = candle::pickle::PthTensors::new(file, None)?;
            let names = if names.is_empty() {
                pth_file
                    .tensor_infos()
                    .keys()
                    .map(|v| v.to_string())
                    .collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match pth_file.get(name)? {
                    Some(tensor) => {
                        println!("{tensor}")
                    }
                    None => println!("not found"),
                }
            }
        }
        Format::Pickle => {
            candle::bail!("pickle format is not supported for print")
        }
        Format::Ggml => {
            let mut file = std::fs::File::open(file)?;
            let content = candle::quantized::ggml_file::Content::read(&mut file, device)?;
            let names = if names.is_empty() {
                content.tensors.keys().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match content.tensors.get(name) {
                    Some(tensor) => {
                        let tensor = tensor.dequantize(device)?;
                        println!("{tensor}")
                    }
                    None => println!("not found"),
                }
            }
        }
        Format::Gguf => {
            let mut file = std::fs::File::open(file)?;
            let content = gguf_file::Content::read(&mut file)?;
            let names = if names.is_empty() {
                content.tensor_infos.keys().map(|v| v.to_string()).collect()
            } else {
                names
            };
            for name in names.iter() {
                println!("==== {name} ====");
                match content.tensor(&mut file, name, device) {
                    Ok(tensor) => {
                        let dtype = tensor.dtype();

                        let tensor = tensor.dequantize(device)?;
                        println!("{tensor} {dtype:?}")
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        println!("not found")
                    }
                }
            }
        }
    }
    Ok(())
}

fn run_ls(
    file: &std::path::PathBuf,
    format: Option<Format>,
    verbose: bool,
    device: &Device,
) -> Result<()> {
    let format = match format {
        Some(format) => format,
        None => match Format::infer(file) {
            Some(format) => format,
            None => {
                println!(
                    "{file:?}: cannot infer format from file extension, use the --format flag"
                );
                return Ok(());
            }
        },
    };
    match format {
        Format::Npz => {
            let tensors = candle::npy::NpzTensors::new(file)?;
            let mut names = tensors.names();
            names.sort();
            for name in names {
                let shape_dtype = match tensors.get_shape_and_dtype(name) {
                    Ok((shape, dtype)) => format!("[{shape:?}; {dtype:?}]"),
                    Err(err) => err.to_string(),
                };
                println!("{name}: {shape_dtype}")
            }
        }
        Format::Safetensors => {
            let tensors = unsafe { candle::safetensors::MmapedSafetensors::new(file)? };
            let mut tensors = tensors.tensors();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, view) in tensors.iter() {
                let dtype = view.dtype();
                let dtype = match candle::DType::try_from(dtype) {
                    Ok(dtype) => format!("{dtype:?}"),
                    Err(_) => format!("{dtype:?}"),
                };
                let shape = view.shape();
                println!("{name}: [{shape:?}; {dtype}]")
            }
        }
        Format::Pth => {
            let mut tensors = candle::pickle::read_pth_tensor_info(file, verbose, None)?;
            tensors.sort_by(|a, b| a.name.cmp(&b.name));
            for tensor_info in tensors.iter() {
                println!(
                    "{}: [{:?}; {:?}]",
                    tensor_info.name,
                    tensor_info.layout.shape(),
                    tensor_info.dtype,
                );
                if verbose {
                    println!("    {:?}", tensor_info);
                }
            }
        }
        Format::Pickle => {
            let file = std::fs::File::open(file)?;
            let mut reader = std::io::BufReader::new(file);
            let mut stack = candle::pickle::Stack::empty();
            stack.read_loop(&mut reader)?;
            for (i, obj) in stack.stack().iter().enumerate() {
                println!("{i} {obj:?}");
            }
        }
        Format::Ggml => {
            let mut file = std::fs::File::open(file)?;
            let content = candle::quantized::ggml_file::Content::read(&mut file, device)?;
            let mut tensors = content.tensors.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, qtensor) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", qtensor.shape(), qtensor.dtype());
            }
        }
        Format::Gguf => {
            let mut file = std::fs::File::open(file)?;
            let content = gguf_file::Content::read(&mut file)?;
            if verbose {
                let mut metadata = content.metadata.into_iter().collect::<Vec<_>>();
                metadata.sort_by(|a, b| a.0.cmp(&b.0));
                println!("metadata entries ({})", metadata.len());
                for (key, value) in metadata.iter() {
                    println!("  {key}: {value:?}");
                }
            }
            let mut tensors = content.tensor_infos.into_iter().collect::<Vec<_>>();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, info) in tensors.iter() {
                println!("{name}: [{:?}; {:?}]", info.shape, info.ggml_dtype);
            }
        }
    }
    Ok(())
}

fn unpack_bitnet_weights(tensor: &Tensor) -> Result<Tensor> {
    let packed_vec = tensor.to_vec2::<u8>().unwrap();

    let rows = tensor.dim(0).unwrap();
    let cols = tensor.dim(1).unwrap();

    let mut unpacked_vec = vec![0f32; rows * 4 * cols];

    for i in 0..rows {
        for j in 0..cols {
            let packed = packed_vec[i][j];

            for k in 0..4 {
                let bits = ((packed >> (k * 2)) & 0b11) as i8 - 1;
                let index = (k * rows + i) * cols + j;
                unpacked_vec[index] = bits as f32;
            }
        }
    }

    let unpacked_tensor = Tensor::from_vec(unpacked_vec, (rows * 4, cols), tensor.device())?;
    Ok(unpacked_tensor)
}

use core::num;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

fn permute(weights: &Tensor, n_head: usize, n_head_kv: Option<usize>) -> Result<Tensor> {
    let n_head = match n_head_kv {
        Some(n_head_kv) if n_head != n_head_kv => n_head_kv,
        _ => n_head,
    };

    let shape = weights.shape();
    let shape0 = shape.dims()[0];
    if shape0 % (n_head * 2) != 0 {
        candle::bail!("weights.shape()[0] is not divisible by (n_head * 2)");
    }

    let mut new_shape = vec![n_head, 2, shape0 / (n_head * 2)];
    new_shape.extend_from_slice(&shape.dims()[1..]);

    let permuted = weights
        .reshape(new_shape)?
        .transpose(1, 2)?
        .reshape(weights.shape())?;

    Ok(permuted)
}

fn run_quantize_safetensors(
    in_files: &[PathBuf],
    out_file: PathBuf,
    q: Quantization,
    bq: Option<Quantization>,
    bitnet_mode: bool,
) -> Result<()> {
    let mut out_file = File::create(out_file)?;
    let dtype = q.dtype();
    let block_size = dtype.block_size();

    let metadata_file = in_files
        .iter()
        .find(|f| f.to_string_lossy().ends_with("config.json"));

    let mut qtensors = Vec::new();

    let mut num_attention_heads = 0;
    let mut num_key_value_heads = 0;
    let mut architecture = String::new();

    let gguf_metadata = if let Some(metadata_file) = metadata_file {
        let metadata_content = std::fs::read_to_string(metadata_file)?;
        let metadata: serde_json::Value = serde_json::from_str(&metadata_content).unwrap();

        num_attention_heads = metadata["num_attention_heads"].as_u64().unwrap();
        num_key_value_heads = metadata["num_key_value_heads"].as_u64().unwrap();
        architecture = metadata["model_type"].as_str().unwrap().to_string();

        vec![
            (
                "llama.attention.head_count",
                gguf_file::Value::from_u32(num_attention_heads as u32),
            ),
            (
                "llama.attention.head_count_kv",
                gguf_file::Value::from_u32(metadata["num_key_value_heads"].as_u64().unwrap() as u32),
            ),
            (
                "llama.block_count",
                gguf_file::Value::from_u32(metadata["num_hidden_layers"].as_u64().unwrap() as u32),
            ),
            (
                "llama.embedding_length",
                gguf_file::Value::from_u32(metadata["hidden_size"].as_u64().unwrap() as u32),
            ),
            (
                "llama.attention.layer_norm_rms_epsilon",
                gguf_file::Value::from_f32(metadata["rms_norm_eps"].as_f64().unwrap() as f32),
            ),
            (
                "llama.rope.dimension_count",
                gguf_file::Value::from_u32(
                    (metadata["hidden_size"].as_u64().unwrap() as u32)
                        / (metadata["num_attention_heads"].as_u64().unwrap() as u32),
                ),
            ),
            (
                "llama.rope.freq_base",
                gguf_file::Value::from_f32(metadata["rope_theta"].as_f64().unwrap() as f32),
            ),
            (
                "general.architecture",
                gguf_file::Value::from_string(architecture.clone()),
            ),
        ]
    } else {
        vec![]
    };
    for in_file in in_files {
        if let Some(metadata) = &metadata_file {
            if Some(in_file) == Some(metadata) {
                continue;
            }
        }

        println!("Loading tensors from file: {:?}", in_file);
        let in_tensors = candle::safetensors::load(in_file, &Device::Cpu)?;

        let processed_tensors = in_tensors
            .into_par_iter()
            .map(|(mut name, tensor)| {
                let mut local_dtype = dtype.clone();
                let mut should_quantize = tensor.rank() == 2 && tensor.dim(1)? % block_size == 0;
                let mut tensor = tensor;

                if should_quantize && bitnet_mode {
                    let is_bitnet_weight = name.contains("self_attn.v_proj")
                        || name.contains("self_attn.q_proj")
                        || name.contains("self_attn.o_proj")
                        || name.contains("self_attn.k_proj")
                        || name.contains("mlp.down_proj")
                        || name.contains("mlp.up_proj")
                        || name.contains("mlp.gate_proj");

                    if is_bitnet_weight {
                        println!("  unpacking {name} {tensor:?} {should_quantize}");
                        tensor = unpack_bitnet_weights(&tensor)?;
                        local_dtype = bq.clone().unwrap().dtype();
                    }
                }

                if name == "lm_head.weight" {
                    local_dtype = GgmlDType::Q6K;
                }

                // apply transformations to the tensors, based on the architecture
                match architecture.as_str() {
                    "llama" => {
                        if name.ends_with("self_attn.q_proj.weight") {
                            tensor = permute(
                                &tensor,
                                num_attention_heads as usize,
                                Some(num_attention_heads as usize),
                            )?;
                        }
                        if name.ends_with("self_attn.k_proj.weight") {
                            tensor = permute(
                                &tensor,
                                num_attention_heads as usize,
                                Some(num_key_value_heads as usize),
                            )?;
                        }
                    }
                    _ => {}
                }

                println!("  quantizing {name} {tensor:?} {should_quantize}");
                let tensor = if should_quantize {
                    QTensor::quantize(&tensor, local_dtype)?
                } else {
                    QTensor::quantize(&tensor, GgmlDType::F32)?
                };

                if name == "model.embed_tokens.weight" {
                    name = "token_embd.weight".to_string();
                } else if name == "model.norm.weight" {
                    name = "output_norm.weight".to_string();
                } else if name == "lm_head.weight" {
                    name = "output.weight".to_string();
                }

                name = name.replace("model.layers.", "blk.");
                name = name.replace("self_attn.q_proj", "attn_q");
                name = name.replace("self_attn.k_proj", "attn_k");
                name = name.replace("self_attn.v_proj", "attn_v");
                name = name.replace("self_attn.o_proj", "attn_output");
                name = name.replace("mlp.gate_proj", "ffn_gate");
                name = name.replace("mlp.down_proj", "ffn_down");
                name = name.replace("mlp.up_proj", "ffn_up");
                name = name.replace("input_layernorm", "attn_norm");
                name = name.replace("post_attention_layernorm", "ffn_norm");

                Ok((name, tensor))
            })
            .collect::<Result<Vec<_>>>()?;

        qtensors.extend(processed_tensors);
    }

    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();

    gguf_file::write(&mut out_file, &gguf_metadata, &qtensors)?;
    Ok(())
}

fn run_dequantize(
    in_file: std::path::PathBuf,
    out_file: std::path::PathBuf,
    device: &Device,
) -> Result<()> {
    let mut in_file = std::fs::File::open(in_file)?;
    let content = gguf_file::Content::read(&mut in_file)?;
    let mut tensors = std::collections::HashMap::new();
    for (tensor_name, _) in content.tensor_infos.iter() {
        let tensor = content.tensor(&mut in_file, tensor_name, device)?;
        let tensor = tensor.dequantize(device)?;
        tensors.insert(tensor_name.to_string(), tensor);
    }
    candle::safetensors::save(&tensors, out_file)?;
    Ok(())
}

fn run_quantize(
    in_files: &[std::path::PathBuf],
    out_file: std::path::PathBuf,
    q: Quantization,
    qmode: QuantizationMode,
    bq: Option<Quantization>,
    bitnet_mode: bool,
    device: &Device,
) -> Result<()> {
    if in_files.is_empty() {
        candle::bail!("no specified input files")
    }
    if bitnet_mode && bq.is_none() {
        candle::bail!("bitnet mode requires a bitnet quantization")
    }
    if let Some(extension) = out_file.extension() {
        if extension == "safetensors" {
            candle::bail!("the generated file cannot use the safetensors extension")
        }
    }
    if let Some(extension) = in_files[0].extension() {
        if extension == "safetensors" {
            return run_quantize_safetensors(in_files, out_file, q, bq, bitnet_mode);
        }
    }

    if in_files.len() != 1 {
        candle::bail!("only a single in-file can be used when quantizing gguf files")
    }

    // Open the out file early so as to fail directly on missing directories etc.
    let mut out_file = std::fs::File::create(out_file)?;
    let mut in_ = std::fs::File::open(&in_files[0])?;
    let content = gguf_file::Content::read(&mut in_)?;
    println!("tensors: {}", content.tensor_infos.len());

    let dtype = q.dtype();
    let qtensors = content
        .tensor_infos
        .par_iter()
        .map(|(name, _)| {
            println!("  quantizing {name}");
            let mut in_file = std::fs::File::open(&in_files[0])?;
            let tensor = content.tensor(&mut in_file, name, device)?;
            let tensor = qmode.quantize(name, tensor, dtype, bitnet_mode)?;
            Ok((name, tensor))
        })
        .collect::<Result<Vec<_>>>()?;
    let qtensors = qtensors
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect::<Vec<_>>();

    let metadata = content
        .metadata
        .iter()
        .map(|(k, v)| (k.as_str(), v.clone()))
        .collect::<Vec<_>>();
    gguf_file::write(&mut out_file, metadata.as_slice(), &qtensors)?;
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let device = Device::Cpu;
    match args.command {
        Command::Ls {
            files,
            format,
            verbose,
        } => {
            let multiple_files = files.len() > 1;
            for file in files.iter() {
                if multiple_files {
                    println!("--- {file:?} ---");
                }
                run_ls(file, format.clone(), verbose, &device)?
            }
        }
        Command::Print {
            file,
            names,
            format,
            full,
            line_width,
        } => run_print(&file, names, format, full, line_width, &device)?,
        Command::Quantize {
            in_file,
            out_file,
            quantization,
            bitnet_quantization,
            mode,
            bitnet_mode,
        } => run_quantize(
            &in_file,
            out_file,
            quantization,
            mode,
            bitnet_quantization,
            bitnet_mode,
            &device,
        )?,
        Command::Dequantize { in_file, out_file } => run_dequantize(in_file, out_file, &device)?,
    }
    Ok(())
}
