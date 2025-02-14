#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle::{DType, Device, Tensor, D};
use candle_nn::{rnn, LSTMConfig, VarBuilder, RNN};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};

const ACCURACY: f32 = 1e-6;

#[derive(Clone, Copy, Debug, clap::ValueEnum, PartialEq, Eq)]
enum WhichModel {
    #[value(name = "lstm")]
    LSTM,
    #[value(name = "gru")]
    GRU,
}

#[derive(Clone, Copy, Debug, Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    cpu: bool,

    #[arg(long, default_value_t = 10)]
    input_dim: usize,

    #[arg(long, default_value_t = 20)]
    hidden_dim: usize,

    #[arg(long, default_value_t = 1)]
    layers: usize,

    #[arg(long)]
    bidirection: bool,

    #[arg(long, default_value_t = 5)]
    batch_size: usize,

    #[arg(long, default_value_t = 3)]
    seq_len: usize,

    #[arg(long, default_value = "lstm")]
    model: WhichModel,

    #[arg(long)]
    test: bool,
}

impl Args {
    pub fn load_model(&self) -> Result<(Config, VarBuilder<'static>, Tensor)> {
        let device = self.device()?;
        if self.test {
            // run unit test and download model from huggingface hub.
            let model = match self.model {
                WhichModel::LSTM => "lstm",
                WhichModel::GRU => "gru",
            };

            let bidirection = if self.bidirection { "bi_" } else { "" };
            let layer = if self.layers > 1 { "_nlayer" } else { "" };
            let model = format!("{}{}{}_test", bidirection, model, layer);
            let (config, vb) = load_model(&model, &device)?;
            let input = vb.get(
                (config.batch_size, config.sequence_length, config.input),
                "input",
            )?;
            Ok((config, vb, input))
        } else {
            let map = candle_nn::VarMap::new();
            let vb = candle_nn::VarBuilder::from_varmap(&map, DType::F32, &device);
            let input = Tensor::randn(
                0.0_f32,
                1.0,
                (self.batch_size, self.seq_len, self.input_dim),
                &device,
            )?;
            Ok((self.into(), vb, input))
        }
    }

    pub fn device(&self) -> Result<Device> {
        Ok(candle_examples::device(self.cpu)?)
    }
}

#[derive(Debug, Clone, PartialEq, serde::Deserialize)]
struct Config {
    pub input: usize,
    pub batch_size: usize,
    pub sequence_length: usize,
    pub hidden: usize,
    pub layers: usize,
    pub bidirection: bool,
}

impl From<&Args> for Config {
    fn from(args: &Args) -> Self {
        Config {
            input: args.input_dim,
            batch_size: args.batch_size,
            sequence_length: args.seq_len,
            hidden: args.hidden_dim,
            layers: args.layers,
            bidirection: args.bidirection,
        }
    }
}

fn load_model(model: &str, device: &Device) -> Result<(Config, VarBuilder<'static>)> {
    let api = Api::new()?;
    let repo_id = "kigichang/test_rnn".to_string();
    let repo = api.repo(Repo::with_revision(
        repo_id,
        RepoType::Model,
        "main".to_string(),
    ));

    let filename = repo.get(&format!("{}.pt", model))?;
    let config_file = repo.get(&format!("{}.json", model))?;

    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let vb = VarBuilder::from_pth(filename, DType::F32, device)?;

    Ok((config, vb))
}

fn assert_tensor(a: &Tensor, b: &Tensor, v: f32) -> Result<()> {
    assert_eq!(a.dims(), b.dims());
    let dim = a.dims().len();
    let mut t = (a - b)?.abs()?;

    for _i in 0..dim {
        t = t.max(D::Minus1)?;
    }

    let t = t.to_scalar::<f32>()?;
    println!("max diff = {}", t);
    assert!(t < v);
    Ok(())
}

fn lstm_config(layer_idx: usize, direction: rnn::Direction) -> LSTMConfig {
    let mut config = LSTMConfig::default();
    config.layer_idx = layer_idx;
    config.direction = direction;
    config
}

fn gru_config(layer_idx: usize, direction: rnn::Direction) -> rnn::GRUConfig {
    let mut config = rnn::GRUConfig::default();
    config.layer_idx = layer_idx;
    config.direction = direction;
    config
}

fn run_lstm(args: Args) -> Result<Tensor> {
    let (config, vb, mut input) = args.load_model()?;

    let mut layers = Vec::with_capacity(config.layers);

    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden
        };
        let lstm_config = lstm_config(layer_idx, rnn::Direction::Forward);
        let lstm = candle_nn::lstm(input_dim, config.hidden, lstm_config, vb.clone())?;
        layers.push(lstm);
    }

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    if args.test {
        let answer = vb.get(
            (config.batch_size, config.sequence_length, config.hidden),
            "output",
        )?;
        assert_tensor(&input, &answer, ACCURACY)?;
    }

    Ok(input)
}

fn run_gru(args: Args) -> Result<Tensor> {
    let (config, vb, mut input) = args.load_model()?;

    let mut layers = Vec::with_capacity(config.layers);

    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden
        };
        let gru_config = gru_config(layer_idx, rnn::Direction::Forward);
        let gru = candle_nn::gru(input_dim, config.hidden, gru_config, vb.clone())?;
        layers.push(gru);
    }

    for layer in &layers {
        let states = layer.seq(&input)?;
        input = layer.states_to_tensor(&states)?;
    }

    if args.test {
        let answer = vb.get(
            (config.batch_size, config.sequence_length, config.hidden),
            "output",
        )?;
        assert_tensor(&input, &answer, ACCURACY)?;
    }

    Ok(input)
}

fn run_bidirectional_lstm(args: Args) -> Result<Tensor> {
    let (config, vb, mut input) = args.load_model()?;

    let mut layers = Vec::with_capacity(config.layers);

    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden * 2
        };

        let forward_config = lstm_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::lstm(input_dim, config.hidden, forward_config, vb.clone())?;

        let backward_config = lstm_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::lstm(input_dim, config.hidden, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.bidirectional_states_to_tensor(&forward_states, &backward_states)?;
    }

    if args.test {
        let answer = vb.get(
            (config.batch_size, config.sequence_length, config.hidden * 2),
            "output",
        )?;
        assert_tensor(&input, &answer, ACCURACY)?;
    }

    Ok(input)
}

fn run_bidirectional_gru(args: Args) -> Result<Tensor> {
    let (config, vb, mut input) = args.load_model()?;

    let mut layers = Vec::with_capacity(config.layers);
    for layer_idx in 0..config.layers {
        let input_dim = if layer_idx == 0 {
            config.input
        } else {
            config.hidden * 2
        };

        let forward_config = gru_config(layer_idx, rnn::Direction::Forward);
        let forward = candle_nn::gru(input_dim, config.hidden, forward_config, vb.clone())?;

        let backward_config = gru_config(layer_idx, rnn::Direction::Backward);
        let backward = candle_nn::gru(input_dim, config.hidden, backward_config, vb.clone())?;

        layers.push((forward, backward));
    }

    for (forward, backward) in &layers {
        let forward_states = forward.seq(&input)?;
        let backward_states = backward.seq(&input)?;
        input = forward.bidirectional_states_to_tensor(&forward_states, &backward_states)?;
    }

    if args.test {
        let answer = vb.get(
            (config.batch_size, config.sequence_length, config.hidden * 2),
            "output",
        )?;
        assert_tensor(&input, &answer, ACCURACY)?;
    }

    Ok(input)
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!(
        "Running {:?} bidirection: {} layers: {} example-test: {}",
        args.model, args.bidirection, args.layers, args.test
    );

    if args.test {
        let test_args = Args {
            model: WhichModel::LSTM,
            bidirection: false,
            layers: 1,
            ..args
        };
        print!("Testing LSTM with 1 layer: ");
        run_lstm(test_args)?;

        let test_args = Args {
            model: WhichModel::GRU,
            bidirection: false,
            layers: 1,
            ..args
        };
        print!("Testing  GRU with 1 layer: ");
        run_gru(test_args)?;

        let test_args = Args {
            model: WhichModel::LSTM,
            bidirection: true,
            layers: 1,
            ..args
        };
        print!("Testing bidirectional LSTM with 1 layer: ");
        run_bidirectional_lstm(test_args)?;

        let test_args = Args {
            model: WhichModel::GRU,
            bidirection: true,
            layers: 1,
            ..args
        };
        print!("Testing bidirectional  GRU with 1 layer: ");
        run_bidirectional_gru(test_args)?;

        let test_args = Args {
            model: WhichModel::LSTM,
            bidirection: false,
            layers: 3,
            ..args
        };
        print!("Testing LSTM with 3 layers: ");
        run_lstm(test_args)?;

        let test_args = Args {
            model: WhichModel::GRU,
            bidirection: false,
            layers: 3,
            ..args
        };
        print!("Testing  GRU with 3 layers: ");
        run_gru(test_args)?;

        let test_args = Args {
            model: WhichModel::LSTM,
            bidirection: true,
            layers: 3,
            ..args
        };
        print!("Testing bidirectional LSTM with 3 layers: ");
        run_bidirectional_lstm(test_args)?;

        let test_args = Args {
            model: WhichModel::GRU,
            bidirection: true,
            layers: 3,
            ..args
        };
        print!("Testing bidirectional  GRU with 3 layers: ");
        run_bidirectional_gru(test_args)?;
    } else {
        let num_directions = if args.bidirection { 2 } else { 1 };
        let batch_size = args.batch_size;
        let seq_len = args.seq_len;
        let hidden_dim = args.hidden_dim;

        let output = match (args.model, args.bidirection) {
            (WhichModel::LSTM, false) => run_lstm(args),
            (WhichModel::GRU, false) => run_gru(args),
            (WhichModel::LSTM, true) => run_bidirectional_lstm(args),
            (WhichModel::GRU, true) => run_bidirectional_gru(args),
        }?;

        assert_eq!(
            output.dims3()?,
            (batch_size, seq_len, hidden_dim * num_directions)
        );
        println!("result dims: {:?}", output.dims());
    }

    Ok(())
}
