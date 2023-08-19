use candle_core::Result;
use clap::{Parser, Subcommand};

#[derive(Subcommand, Debug, Clone)]
enum Command {
    Ls { files: Vec<std::path::PathBuf> },
}

#[derive(Parser, Debug, Clone)]
struct Args {
    /// Enable verbose mode.
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Command,
}

fn run_ls(file: &std::path::PathBuf) -> Result<()> {
    match file.extension().and_then(|e| e.to_str()) {
        Some("npz") => {
            let tensors = candle_core::npy::NpzTensors::new(file)?;
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
        Some("safetensor") | Some("safetensors") => {
            let tensors = unsafe { candle_core::safetensors::MmapedFile::new(file)? };
            let tensors = tensors.deserialize()?;
            let mut tensors = tensors.tensors();
            tensors.sort_by(|a, b| a.0.cmp(&b.0));
            for (name, view) in tensors.iter() {
                let dtype = view.dtype();
                let dtype = match candle_core::DType::try_from(dtype) {
                    Ok(dtype) => format!("{dtype:?}"),
                    Err(_) => format!("{dtype:?}"),
                };
                let shape = view.shape();
                println!("{name}: [{shape:?}; {dtype}]")
            }
        }
        Some(_) => {
            println!("{file:?}: unsupported file extension")
        }
        None => {
            println!("{file:?}: no file extension")
        }
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Ls { files } => {
            let multiple_files = files.len() > 1;
            for file in files.iter() {
                if multiple_files {
                    println!("--- {file:?} ---");
                }
                run_ls(file)?
            }
        }
    }
    Ok(())
}
