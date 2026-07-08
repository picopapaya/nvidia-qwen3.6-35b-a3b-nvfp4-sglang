# nvidia-qwen3.6-35b-a3b-nvfp4-sglang

Docker image that runs **NVIDIA's NVFP4 quantization of Qwen3.6-35B-A3B** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The weights are pre-quantized to **NVFP4** by NVIDIA with [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and served with [SGLang](https://github.com/sgl-project/sglang). The checkpoint is ModelOpt **mixed precision** (NVFP4 experts + FP8 linear-attention projections), so no `--quantization` flag is passed — SGLang auto-detects it as `modelopt_mixed`.

## What it is

Qwen3.6-35B-A3B is a hybrid-attention Mixture-of-Experts vision-language model:

- 35 billion total parameters, ~3 billion active per token (256 experts per MoE layer, 8 routed + 1 shared active), so it runs with the compute cost of a much smaller model.
- Hybrid attention: 3 of every 4 layers use linear attention (Gated DeltaNet), every 4th layer uses full attention — long contexts are cheap in both compute and cache memory.
- Multimodal: accepts text and images.
- 262,144-token native context.

## NVFP4 on the GB10

This chip has no fast native path for 4-bit math, so SGLang has to unpack NVFP4 back to a bigger format (BF16) on the fly before it can compute with it — in theory, that should erase any speed advantage. In practice, tested directly on this box, it doesn't — see below. What NVFP4 buys regardless of speed:

- **~20 GB weights** instead of ~70 GB for the unquantized model — a much faster download and far more memory left free for the KV cache (the model's working memory) and other models.
- NVIDIA's own carefully-calibrated quantization, rather than a generic on-the-fly conversion.
- The linear-attention layers are kept at a higher precision (8-bit) by NVIDIA's tool, protecting the parts of the model most sensitive to rounding error.

### Measured performance (2026-07-08, one request at a time, this box)

| Config | tokens/sec | vs FP8 (60.5 tok/s) |
|---|---|---|
| This image, default settings (`triton`) | 65.6 | 8% faster |
| This image, alternate kernel (`flashinfer`) | 65.3 | 8% faster (switching kernels made no real difference) |

This **contradicts an earlier note in the FP8 image's docs** claiming FP8 was faster — that note was based on other people's online benchmarks, not testing on this machine, and it didn't hold up (corrected in `../qwen-qwen3.6-35b-a3b-fp8-sglang/README.md`). What we haven't checked yet:

- Several requests running at once — we only tested one at a time so far, and that's not how this box is actually used day to day.
- Whether NVFP4's answers are as accurate as FP8's — going down to a smaller number format can lose some precision, and we haven't measured how much that matters here.
- Switching to the `flashinfer` kernel made no difference for this image — unlike the FP8 image, where it was worth a real ~30% speedup. The slow part here is likely the "unpacking NVFP4" step, not the attention math itself.

### MTP (a speed feature) — not available, not sure if it even can be

MTP lets the model predict several words at once instead of one at a time, speeding things up a lot when it works. We don't have it wired up on this image (no on/off switch exists for it here). As of 2026-07-08:

- NVIDIA's own page for this exact model says MTP works with it — but only through a different serving engine (vLLM), not the one we use here (SGLang).
- SGLang's own documentation currently says NVFP4 is only officially supported for a smaller version of this model (27B, not our 35B version) — we're already off the beaten path just getting this image to run at all (that's why it needs a special nightly build with patches — see "SGLang compatibility" below).
- Bottom line: we honestly don't know if MTP is even possible with our current setup. Nobody has confirmed it one way or the other for SGLang specifically. Worth checking again if a newer SGLang version comes out.

## SGLang compatibility

SGLang **v0.5.13+** is required for the `qwen3_5_moe` architecture, and CUDA 13.x for sm_121a — but the latest release (v0.5.14) still cannot load this checkpoint:

1. It routes ModelOpt `MIXED_PRECISION` checkpoints to its DeepSeek-oriented `w4afp8` loader, which hardcodes 128×128 FP8 weight blocks and crashes on this model's 32-wide linear-attention projections (`output_partition_size = 32 is not divisible by ... block_n = 128`).
2. Even with the routing fixed, its MoE loader miscomputes shard sizes for NVFP4-packed expert weights (`start (0) + length (512) exceeds dimension size (256)`).

Both are fixed on SGLang main, so this image pins a main-branch nightly (`lmsysorg/sglang:nightly-dev-cu13-20260707-b4155233`) and additionally carries `patches/modelopt-mixed-routing.py`, an idempotent backport of the routing fix that no-ops on patched bases but protects if the base is ever moved back to a release tag. Switch to a stable release tag once one ships these fixes.

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

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token (avoids anonymous rate limits) |
| `QUANTIZATION` | `auto` | `auto`/empty lets SGLang detect from the checkpoint (`modelopt_mixed`); any other value is passed as `--quantization` |
| `CONTEXT_LEN` | `262144` | Maximum context length in tokens |
| `MEM_FRACTION` | `0.85` | Fraction of VRAM reserved for weights + KV cache |
| `MAX_RUNNING_REQUESTS` | `4` | Maximum concurrent requests |
| `REASONING_PARSER` | `qwen3` | SGLang reasoning parser |
| `TOOL_CALL_PARSER` | `qwen3_coder` | SGLang tool-call parser (per the SGLang Qwen3.6 cookbook) |
| `ATTENTION_BACKEND` | `triton` | Attention backend for the full-attention layers. Now exposed as a compose override (2026-07-08); `flashinfer` measured no faster than the `triton` default for this image (see "Measured performance" above) |
| `EXTRA_ARGS` | *(empty)* | Extra flags passed directly to `sglang.launch_server` |

## LiteLLM router

The model is registered in the shared LiteLLM proxy (`/home/shared/Documents/litellm/config.yaml`) as `nvidia-qwen3.6-35b-a3b-nvfp4-sglang`. Restart the router after config changes:

```bash
docker restart litellm
```

## Publishing

Pushing a `v*.*.*` tag to GitHub builds the linux/arm64 image and publishes it to Docker Hub as `picopapaya/nvidia-qwen3.6-35b-a3b-nvfp4-sglang` (see `.github/workflows/docker-publish.yml`; requires `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets).

## License

MIT — see [LICENSE](LICENSE).
