# candle-clip

Contrastive Language-Image Pre-Training (CLIP) is an architecture trained on
pairs of images with related texts.

https://github.com/openai/CLIP

https://github.com/huggingface/transformers/tree/f6fa0f0bf0796ac66f201f23bdb8585de1609add/src/transformers/models/clip

## Running on an example on cpu

```
$ cargo run --example clip --release -- --images "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg","candle-examples/examples/yolo-v8/assets/bike.jpg" --cpu --sequences  "a cycling race","a photo of two cats","a robot holding a candle"


> Results for image: candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg

> INFO clip: Probability: 0.0000% Text: a cycling race
> INFO clip: Probability: 0.0000% Text: a photo of two cats
> INFO clip: Probability: 100.0000% Text: a robot holding a candle

> Results for image: candle-examples/examples/yolo-v8/assets/bike.jpg

> INFO clip: Probability: 99.9999% Text: a cycling race
> INFO clip: Probability: 0.0001% Text: a photo of two cats
> INFO clip: Probability: 0.0000% Text: a robot holding a candle
```

### Arguments
- `--model`: local path to the model. If not provided will download from huggingface
- `--tokenizer`: local path to the tokenizer.json file. If not provided will download from huggingface
- `--images`: list of images to use. 

Example: `--images candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg,candle-examples/examples/yolo-v8/assets/bike.jpg`
- `sequences`: list of text sequences to use. 

Example: `--sequences "a cycling race","bike"`

- `--model-id`: model id to use from huggingface. Example: `--model-id openai/clip-vit-large-patch14`
- `--revision`: revision to use from huggingface. Example: `--revision refs/pr/4`
- `--use-pth`: Use the pytorch weights rather than the safetensors ones. Default: true
- `--cpu`: Use cpu. Use `--cpu false` for gpu but requires gpu support with `--features cuda` 

## Running on an example with metal feature (mac)

```
$ cargo run --features metal --example clip --release -- --images "candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg","candle-examples/examples/yolo-v8/assets/bike.jpg" --cpu --sequences "a cycling race","a photo of two cats","a robot holding a candle"


Results for image: candle-examples/examples/stable-diffusion/assets/stable-diffusion-xl.jpg

INFO clip: Probability: 0.0000% Text: a cycling race
INFO clip: Probability: 0.0000% Text: a photo of two cats
INFO clip: Probability: 100.0000% Text: a robot holding a candle

Results for image: candle-examples/examples/yolo-v8/assets/bike.jpg

INFO clip: Probability: 99.9999% Text: a cycling race
INFO clip: Probability: 0.0001% Text: a photo of two cats
INFO clip: Probability: 0.0000% Text: a robot holding a candle
```
