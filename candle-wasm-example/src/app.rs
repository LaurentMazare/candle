use crate::console_log;
use crate::worker::{ModelData, Worker, WorkerInput, WorkerOutput};
use js_sys::Date;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use yew::{html, Component, Context, Html};
use yew_agent::{Bridge, Bridged};

const SAMPLE_NAMES: [&str; 6] = [
    "jfk.wav", "a13.wav", "gb0.wav", "gb1.wav", "hp0.wav", "mm0.wav",
];

async fn fetch_url(url: &str) -> Result<Vec<u8>, JsValue> {
    use web_sys::{Request, RequestCache, RequestInit, RequestMode, Response};
    let window = web_sys::window().ok_or("window")?;
    let mut opts = RequestInit::new();
    let opts = opts
        .method("GET")
        .mode(RequestMode::Cors)
        .cache(RequestCache::NoCache);

    let request = Request::new_with_str_and_init(url, opts)?;

    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;

    // `resp_value` is a `Response` object.
    assert!(resp_value.is_instance_of::<Response>());
    let resp: Response = resp_value.dyn_into()?;
    let data = JsFuture::from(resp.blob()?).await?;
    let blob = web_sys::Blob::from(data);
    let array_buffer = JsFuture::from(blob.array_buffer()).await?;
    let data = js_sys::Uint8Array::new(&array_buffer).to_vec();
    Ok(data)
}

pub enum Msg {
    Run(usize),
    UpdateStatus(String),
    SetDecoder(ModelData),
    WorkerInMsg(WorkerInput),
    WorkerOutMsg(WorkerOutput),
}

pub struct App {
    status: String,
    content: String,
    decode_in_flight: bool,
    worker: Box<dyn Bridge<Worker>>,
}

async fn model_data_load() -> Result<ModelData, JsValue> {
    let tokenizer = fetch_url("tokenizer.en.json").await?;
    let mel_filters = fetch_url("mel_filters.safetensors").await?;
    let weights = fetch_url("tiny.en.safetensors").await?;
    console_log!("{}", weights.len());
    Ok(ModelData {
        tokenizer,
        mel_filters,
        weights,
    })
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let status = "loading weights".to_string();
        let cb = {
            let link = ctx.link().clone();
            move |e| link.send_message(Self::Message::WorkerOutMsg(e))
        };
        let worker = Worker::bridge(std::rc::Rc::new(cb));
        Self {
            status,
            content: String::new(),
            decode_in_flight: false,
            worker,
        }
    }

    fn rendered(&mut self, ctx: &Context<Self>, first_render: bool) {
        if first_render {
            ctx.link().send_future(async {
                match model_data_load().await {
                    Err(err) => {
                        let status = format!("{err:?}");
                        Msg::UpdateStatus(status)
                    }
                    Ok(model_data) => Msg::SetDecoder(model_data),
                }
            });
        }
    }

    fn update(&mut self, ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetDecoder(md) => {
                self.status = "weights loaded succesfully!".to_string();
                console_log!("loaded weights");
                self.worker.send(WorkerInput::ModelData(md));
                true
            }
            Msg::Run(sample_index) => {
                let sample = SAMPLE_NAMES[sample_index];
                if self.decode_in_flight {
                    self.content = "already decoding some sample at the moment".to_string()
                } else {
                    self.decode_in_flight = true;
                    self.status = format!("decoding {sample}");
                    self.content = String::new();
                    ctx.link().send_future(async move {
                        match fetch_url(sample).await {
                            Err(err) => {
                                let value = Err(format!("decoding error: {err:?}"));
                                // Mimic a worker output to so as to release decode_in_flight
                                Msg::WorkerOutMsg(WorkerOutput { value })
                            }
                            Ok(wav_bytes) => {
                                Msg::WorkerInMsg(WorkerInput::DecodeTask { wav_bytes })
                            }
                        }
                    })
                }
                //
                true
            }
            Msg::WorkerOutMsg(WorkerOutput { value }) => {
                self.status = "Worker responded!".to_string();
                self.content = format!("{value:?}");
                self.decode_in_flight = false;
                true
            }
            Msg::WorkerInMsg(inp) => {
                self.worker.send(inp);
                true
            }
            Msg::UpdateStatus(status) => {
                self.status = status;
                true
            }
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <div>
                <table>
                <thead>
                <tr>
                  <th>{"Sample"}</th>
                  <th></th>
                  <th></th>
                </tr>
                </thead>
                <tbody>
                {
                    SAMPLE_NAMES.iter().enumerate().map(|(i, name)| { html! {
                <tr>
                  <th>{name}</th>
                  <th><audio controls=true src={format!("./{name}")}></audio></th>
                  <th><button class="button" onclick={ctx.link().callback(move |_| Msg::Run(i))}> { "run" }</button></th>
                </tr>
                    }
                    }).collect::<Html>()
                }
                </tbody>
                </table>
                <h2>
                  {&self.status}
                </h2>
                {
                    if self.decode_in_flight {
                        html! { <progress id="progress-bar" aria-label="decoding…"></progress> }
                    } else { html!{
                <blockquote>
                <p>
                  {&self.content}
                </p>
                </blockquote>
                }
                }
                }

                // Display the current date and time the page was rendered
                <p class="footer">
                    { "Rendered: " }
                    { String::from(Date::new_0().to_string()) }
                </p>
            </div>
        }
    }
}
