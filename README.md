# nvidia-qwen3.6-35b-a3b-nvfp4-sglang

Docker image that runs **NVIDIA's NVFP4 quantization of Qwen3.6-35B-A3B** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The weights are pre-quantized to **NVFP4** by NVIDIA with [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and served with [SGLang](https://github.com/sgl-project/sglang) via `--quantization modelopt_fp4`.

## What it is

Qwen3.6-35B-A3B is a hybrid-attention Mixture-of-Experts vision-language model:

- 35 billion total parameters, ~3 billion active per token (256 experts per MoE layer, 8 routed + 1 shared active), so it runs with the compute cost of a much smaller model.
- Hybrid attention: 3 of every 4 layers use linear attention (Gated DeltaNet), every 4th layer uses full attention — long contexts are cheap in both compute and cache memory.
- Multimodal: accepts text and images.
- 262,144-token native context.

## NVFP4 on the GB10

The GB10 (SM_121a) has no native FP4 GEMM kernel, so SGLang serves NVFP4 through the Marlin kernel, which dequantizes FP4 → BF16 on the fly — the FP4 compute speedup is lost. What NVFP4 still buys on this machine:

- **~20 GB weights** instead of ~70 GB BF16 — a much faster first download and far more unified memory left for KV cache and other models.
- NVIDIA's calibrated quantization scales (ModelOpt), rather than on-the-fly quantization.
- Linear-attention projections are kept at 8-bit by ModelOpt (mixed precision), protecting the recurrent-state layers from FP4 error.

SGLang **v0.5.13+** is required for the `qwen3_5_moe` architecture; this image uses `lmsysorg/sglang:v0.5.14-cu130` (CUDA 13.x is required for sm_121a).

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
| `CONTEXT_LEN` | `262144` | Maximum context length in tokens |
| `MEM_FRACTION` | `0.85` | Fraction of VRAM reserved for weights + KV cache |
| `MAX_RUNNING_REQUESTS` | `4` | Maximum concurrent requests |
| `REASONING_PARSER` | `qwen3` | SGLang reasoning parser |
| `TOOL_CALL_PARSER` | `qwen3_coder` | SGLang tool-call parser (per the SGLang Qwen3.6 cookbook) |
| `ATTENTION_BACKEND` | `triton` | Attention backend for the full-attention layers |
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
