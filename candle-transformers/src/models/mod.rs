//! Candle Implementaitons of many Transformer models
//!
//! This crate provides implementations of popular machine learning models and architectures.
//! These include:
//! * Transformer models like [`bert`], [`t5`], [`llama`], [`mistral`] etc.
//! * Vision models like [`resnet`], [`vit`], [`convnext`], etc.
//! * Audio models like [`whisper`].
//! * Multi-modal models like [`blip`]  and [`llava`].
//!
//! The implementations aim to be readable while maintaining good performance.

pub mod based;
pub mod beit;
pub mod bert;
pub mod bigcode;
pub mod blip;
pub mod blip_text;
pub mod chatglm;
pub mod chinese_clip;
pub mod clip;
pub mod codegeex4_9b;
pub mod colpali;
pub mod convmixer;
pub mod convnext;
pub mod dac;
pub mod depth_anything_v2;
pub mod dinov2;
pub mod dinov2reg4;
pub mod distilbert;
pub mod efficientnet;
pub mod efficientvit;
pub mod encodec;
pub mod eva2;
pub mod falcon;
pub mod fastvit;
pub mod flux;
pub mod gemma;
pub mod gemma2;
pub mod glm4;
pub mod granite;
pub mod hiera;
pub mod jina_bert;
pub mod llama;
pub mod llama2_c;
pub mod llama2_c_weights;
pub mod llava;
pub mod mamba;
pub mod marian;
pub mod metavoice;
pub mod mimi;
pub mod mistral;
pub mod mixformer;
pub mod mixtral;
pub mod mmdit;
pub mod mobileclip;
pub mod mobilenetv4;
pub mod mobileone;
pub mod moondream;
pub mod mpt;
pub mod olmo;
pub mod openclip;
pub mod paligemma;
pub mod parler_tts;
pub mod persimmon;
pub mod phi;
pub mod phi3;
pub mod pixtral;
pub mod quantized_blip;
pub mod quantized_blip_text;
pub mod quantized_llama;
pub mod quantized_llama2_c;
pub mod quantized_metavoice;
pub mod quantized_mistral;
pub mod quantized_mixformer;
pub mod quantized_moondream;
pub mod quantized_mpt;
pub mod quantized_phi;
pub mod quantized_phi3;
pub mod quantized_qwen2;
pub mod quantized_recurrent_gemma;
pub mod quantized_rwkv_v5;
pub mod quantized_rwkv_v6;
pub mod quantized_stable_lm;
pub mod quantized_t5;
pub mod qwen2;
pub mod qwen2_moe;
pub mod recurrent_gemma;
pub mod repvgg;
pub mod resnet;
pub mod rwkv_v5;
pub mod rwkv_v6;
pub mod segformer;
pub mod segment_anything;
pub mod siglip;
pub mod stable_diffusion;
pub mod stable_lm;
pub mod starcoder2;
pub mod stella_en_v5;
pub mod t5;
pub mod trocr;
pub mod vgg;
pub mod vit;
pub mod whisper;
pub mod with_tracing;
pub mod wuerstchen;
pub mod yi;
