# nvidia-qwen3.6-35b-a3b-nvfp4-sglang

Docker image that runs **NVIDIA's NVFP4 quantization of Qwen3.6-35B-A3B** as an OpenAI-compatible API server, built for the **NVIDIA GB10 (DGX Spark)**.

The weights are pre-quantized to **NVFP4** by NVIDIA with [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer) and served with [SGLang](https://github.com/sgl-project/sglang). The checkpoint mixes precisions — most of the model is NVFP4, but a sensitive part (the linear-attention layers) is kept at a higher precision (FP8) to protect accuracy. SGLang detects this automatically; no `--quantization` flag needs to be set by hand.

## What this image is

Qwen3.6-35B-A3B is a Mixture-of-Experts model that also understands images, not just text:

- 35 billion total parameters, but for any given word it only actually uses about 3 billion of them ("35B-A3B" = 35B total, ~3B active). The rest sit in memory ready to be picked, but don't add to the compute cost — it runs like a much smaller model while still having the knowledge of a much bigger one.
- Most of its layers use a cheaper, memory-light form of attention (a technique called Gated DeltaNet); only 1 in every 4 layers uses the full, more expensive kind. This is why it can handle very long conversations without needing huge amounts of extra memory for each one.
- Accepts both text and images.
- Supports up to 262,144 tokens (roughly 200,000 words) of context.

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

1. It routes this kind of mixed-precision checkpoint to a loader built for a different model family, which makes assumptions about weight shapes that don't hold here and crashes.
2. Even after that's worked around, its MoE loader gets the memory layout wrong for NVFP4-packed weights.

Both are fixed on SGLang's main development branch, so this image pins a specific nightly build (`lmsysorg/sglang:nightly-dev-cu13-20260707-b4155233`) rather than an official release, plus a small patch that backports the fix. Switch to a stable release tag once one ships these fixes — check back periodically.

## Configuration

### Tunable via `.env`

These have a default baked into the image, but you can override them per-deployment by setting them in a `.env` file next to `docker-compose.yml`. Docker Compose reads that file automatically and passes the values into the container when it starts — no image rebuild needed, just edit `.env` and restart.

| Variable | Default | What it does |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Optional Hugging Face token — avoids download rate limits, not required (this model isn't gated) |
| `CONTEXT_LEN` | `262144` | The longest conversation/prompt (in tokens) the server will accept |
| `MEM_FRACTION` | `0.85` | How much of the GPU's memory this server is allowed to claim |
| `ATTENTION_BACKEND` | `triton` | Which kernel library handles the attention math — measured no faster with `flashinfer` for this image (see "Measured performance" above) |

### Fixed — not overridable via `.env`

These define what this image *is*, not how it's tuned. Changing them means you're describing a different image, not adjusting this one.

| Variable | Value | Why it's fixed |
|---|---|---|
| `MODEL_ID` | `nvidia/Qwen3.6-35B-A3B-NVFP4` | This is which model the image downloads and runs — that's the image's whole identity |
| `QUANTIZATION` | `auto` | Left as "auto" so SGLang detects the mixed-precision format itself; not something to tune |
| `KV_CACHE_DTYPE` | `auto` | Left to SGLang to pick automatically |
| `MAX_RUNNING_REQUESTS` | `4` | Not currently wired up as a `.env` override — could be added if a need for it comes up |
| `REASONING_PARSER` | `qwen3` | Needed so SGLang understands this model's "thinking" output format |
| `TOOL_CALL_PARSER` | `qwen3_coder` | Needed so SGLang understands this model's function-calling output format |

`EXTRA_ARGS` also exists (passed straight through to the underlying server command) but isn't wired to `.env` by default — it's commented out in `docker-compose.yml` as a documented escape hatch. Uncomment it there directly if you need to pass something not covered above.

## Running alongside another ~30B-class model

The packaged defaults (`CONTEXT_LEN=262144`, `MEM_FRACTION=0.85`) assume this is the only large model on the GPU. To run it side by side with another ~30B-class model (e.g. the FP8 sibling image) on a roughly 50/50 memory split, set in `.env`:

```
CONTEXT_LEN=131072
MEM_FRACTION=0.5
```

**Verified 2026-07-08:** both this image and the FP8 sibling started and ran healthy at the same time with these settings, and 4 concurrent requests at realistic prompt sizes (~7-8K tokens) all completed cleanly with no errors.

**This image handles the split better than the FP8 sibling.** Its smaller weights (~22 GB) leave a much bigger KV cache pool at these settings — **277,600 tokens**, vs FP8's 102,798 — enough for roughly two fully-independent 128K-context requests (FP8's pool can't even hold one). `MAX_RUNNING_REQUESTS` stays at the full 4 here, no auto-reduction needed. Still short of 3-4 truly independent max-context requests, but comfortably handles 4 concurrent requests at realistic prompt sizes like the ones tested above — treat `CONTEXT_LEN` as a safety ceiling for your longest realistic prompt, not a guarantee of full-context concurrency.

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

## LiteLLM router

The model is registered in the shared LiteLLM proxy (`/home/shared/Documents/litellm/config.yaml`) as `nvidia-qwen3.6-35b-a3b-nvfp4-sglang`. Restart the router after config changes:

```bash
docker restart litellm
```

## Publishing

Pushing a `v*.*.*` tag to GitHub builds the linux/arm64 image and publishes it to Docker Hub as `picopapaya/nvidia-qwen3.6-35b-a3b-nvfp4-sglang` (see `.github/workflows/docker-publish.yml`; requires `DOCKERHUB_USERNAME` / `DOCKERHUB_TOKEN` repo secrets).

## License

MIT — see [LICENSE](LICENSE).
