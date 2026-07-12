# nvidia-qwen3.6-35b-a3b-nvfp4-sglang

Docker image that runs **NVIDIA's NVFP4 quantization of Qwen3.6-35B-A3B** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The weights are pre-quantized to **NVFP4** by NVIDIA with [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and served with [SGLang](https://github.com/sgl-project/sglang). The checkpoint mixes precisions — most of the model is NVFP4, but a sensitive part (the linear-attention layers) is kept at a higher precision (FP8) to protect accuracy. SGLang detects this automatically; no `--quantization` flag needs to be set by hand.

## What this image is

Qwen3.6-35B-A3B is a Mixture-of-Experts model that also understands images, not just text:

- 35 billion total parameters, but for any given word it only actually uses about 3 billion of them ("35B-A3B" = 35B total, ~3B active). The rest sit in memory ready to be picked, but don't add to the compute cost — it runs like a much smaller model while still having the knowledge of a much bigger one.
- Most of its layers use a cheaper, memory-light form of attention (a technique called Gated DeltaNet); only 1 in every 4 layers uses the full, more expensive kind. This is why it can handle very long conversations without needing huge amounts of extra memory for each one.
- Accepts both text and images.
- Supports up to 262,144 tokens (roughly 200,000 words) of context.

### NVFP4 on the GB10

This chip has no fast native path for 4-bit math, so SGLang has to unpack NVFP4 back to a bigger format (BF16) on the fly before it can compute with it. What NVFP4 buys regardless of that conversion step:

- **~20 GB weights** instead of ~70 GB for the unquantized model — a much faster download and far more memory left free for the KV cache (the model's working memory) and other models.
- NVIDIA's own carefully-calibrated quantization, rather than a generic on-the-fly conversion.
- The linear-attention layers are kept at a higher precision (8-bit) by NVIDIA's tool, protecting the parts of the model most sensitive to rounding error.

Compare with [`qwen-qwen3.6-35b-a3b-fp8-sglang`](https://github.com/picopapaya/qwen-qwen3.6-35b-a3b-fp8-sglang), which serves the Qwen team's own FP8 quantization of the same model. See `EXPERIMENT_NOTES.md` for measured speed comparisons between the two on this box.

### SGLang compatibility

SGLang **v0.5.13+** is required for the `qwen3_5_moe` architecture, and CUDA 13.x for sm_121a. This image pins [`lmsysorg/sglang:v0.5.15-cu130`](https://github.com/sgl-project/sglang/releases), the first official SGLang release with native loading support for this checkpoint's mixed-precision (ModelOpt) format — no quantization-loading patch is needed.

One patch remains: a generic SGLang bug in the [CUDA-graph buffer sizing for MTP/EAGLE target-verify](https://docs.sglang.io/advanced_features/speculative_decoding.html) under-allocates on the triton attention backend, crashing the scheduler once a batch's context usage crosses a threshold. See `patches/triton-target-verify-mask-buffer.py` and `EXPERIMENT_NOTES.md` for the root cause.

[MTP speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html) (patched, see above) is enabled by default via `EXTRA_ARGS` — see `EXPERIMENT_NOTES.md` for measured throughput gains.

## Configuration

### Tunable via `.env`

These have a default baked into the image, but you can override them per-deployment by setting them in a `.env` file next to `docker-compose.yml`. Docker Compose reads that file automatically and passes the values into the container when it starts — no image rebuild needed, just edit `.env` and restart.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Optional [Hugging Face token](https://huggingface.co/docs/hub/security-tokens) — avoids download rate limits, not required (this model isn't gated) |
| `CONTEXT_LEN` | `262144` | The longest conversation/prompt (in tokens) the server will accept — SGLang's [`--context-length`](https://docs.sglang.io/advanced_features/server_arguments.html#model-and-tokenizer) |
| `MEM_FRACTION` | `0.85` | How much of the GPU's memory this server is allowed to claim — SGLang's [`--mem-fraction-static`](https://docs.sglang.io/advanced_features/server_arguments.html#memory-and-scheduling) |
| `ATTENTION_BACKEND` | `triton` | Which kernel library handles the [attention math](https://docs.sglang.io/advanced_features/attention_backend.html) |
| `EXTRA_ARGS` | `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-fused-qk-norm-rope` | Extra flags passed straight to the SGLang server command. The default turns on [MTP speculative decoding](https://docs.sglang.io/advanced_features/speculative_decoding.html) plus a fused QK-norm-RoPE kernel for faster decoding (patched, see "SGLang compatibility" above) — see `EXPERIMENT_NOTES.md` for measured gains. Override to pass something else, or to add flags like `--cuda-graph-max-bs` |

### Fixed — not overridable via `.env`

These define what this image *is*, not how it's tuned. Changing them means you're describing a different image, not adjusting this one.

| Variable | Value | Why it's fixed |
|---|---|---|
| `MODEL_ID` | [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) | This is which model the image downloads and runs — that's the image's whole identity |
| `QUANTIZATION` | `auto` | Left as "auto" so SGLang [detects the mixed-precision format](https://docs.sglang.io/advanced_features/server_arguments.html#quantization-and-data-type) itself; not something to tune |
| `KV_CACHE_DTYPE` | `auto` | Left to [SGLang to pick automatically](https://docs.sglang.io/advanced_features/server_arguments.html#quantization-and-data-type) |
| `MAX_RUNNING_REQUESTS` | `4` | SGLang's [`--max-running-requests`](https://docs.sglang.io/advanced_features/server_arguments.html#memory-and-scheduling) — not currently wired up as a `.env` override; could be added if a need for it comes up |
| `REASONING_PARSER` | `qwen3` | Needed so SGLang understands this model's ["thinking" output format](https://docs.sglang.io/advanced_features/server_arguments.html#api-related) |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Needed so SGLang understands this model's [function-calling output format](https://docs.sglang.io/advanced_features/server_arguments.html#api-related) |

## Requirements

- NVIDIA GB10 / DGX Spark (SM_121a)
- Docker with NVIDIA Container Toolkit
- The `llm-net` Docker network: `docker network create llm-net`
- A Hugging Face token is **optional** — the model is not gated (Apache-2.0)

## Usage

```bash
# Prod — pull image from Docker Hub
docker compose up

# Dev — build image locally
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The server starts on port **30000** and exposes an OpenAI-compatible API once the health check passes (allow up to 10 minutes for the first run while weights download).

### LiteLLM router

The model is registered in the shared LiteLLM proxy (`/home/shared/Documents/litellm/config.yaml`) as `nvidia-qwen3.6-35b-a3b-nvfp4-sglang`. Restart the router after config changes:

```bash
docker restart litellm
```

### Publishing

Pushing a `v*.*.*` tag to GitHub builds the linux/arm64 image and publishes it to Docker Hub as `picopapaya/nvidia-qwen3.6-35b-a3b-nvfp4-sglang` (see `.github/workflows/docker-publish.yml`; requires `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets).

## License

MIT — see [LICENSE](LICENSE).
