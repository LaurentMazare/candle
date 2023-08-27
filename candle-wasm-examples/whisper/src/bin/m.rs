use candle_wasm_example_whisper::worker::{Decoder as D, ModelData, WorkerOutput};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Decoder {
    decoder: D,
}

#[wasm_bindgen]
impl Decoder {
    #[wasm_bindgen(constructor)]
    pub fn new(
        weights: Vec<u8>,
        tokenizer: Vec<u8>,
        mel_filters: Vec<u8>,
    ) -> Result<Decoder, JsError> {
        let decoder = D::load(ModelData {
            tokenizer,
            mel_filters,
            weights,
        });

        match decoder {
            Ok(decoder) => Ok(Self { decoder }),
            Err(e) => Err(JsError::new(&e.to_string())),
        }
    }

    #[wasm_bindgen]
    pub fn decode(&self, wav_input: Vec<u8>) -> Result<String, JsError> {
        let segments = self
            .decoder
            .convert_and_run(&wav_input)
            .map(WorkerOutput::Decoded)
            .map_err(|e| e.to_string());

        let json = serde_json::to_string(&segments)?;
        Ok(json)
    }
}

fn main() {}
