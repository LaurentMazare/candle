# DeepSeek V2

DeepSeek V2 an MoE model featuring MLA (Multi-Latent Attention). There is a lite (16B) and a full (236B) model.

- Context length of **32k tokens** (Lite model), **128k tokens** (full model)
- 64 routed experts (Lite model), 160 routed experts (full model)

## Running the example

```bash
$ cargo run --example deepseekv2 --release --features metal -- --prompt "Recursive fibonacci code in Rust:" --which lite --sample-len 150  

fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        return n;
    } else {
        return fibonacci(n - 1) + fibonacci(n - 2);
    }
}

## Fibonacci code in Python:

def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

## Fibonacci code in JavaScript:

function fibonacci(n) {
    if (n <= 1
```
