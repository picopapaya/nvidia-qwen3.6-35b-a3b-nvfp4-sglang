# NVIDIA Qwen3.6-35B-A3B pre-quantized to NVFP4 (ModelOpt) served by SGLang, on the NVIDIA GB10 (DGX Spark).
#
# Model architecture — Qwen3.6-35B-A3B is a hybrid-attention Mixture-of-Experts VLM:
#   - 35B total parameters, ~3B active per token ("35B-A3B" = 35B total, 3B active)
#   - 256 experts per MoE layer; router activates 8 per token (+ shared expert)
#   - Hybrid: 3 of every 4 layers use linear attention (Gated DeltaNet), 1 uses full attention
#   - Multimodal: includes a vision encoder (text + image input)
#   - 262144-token native context (rope_theta 1e7)
#
# Quantization — NVFP4 on this hardware (GB10):
#   - This chip has no fast native path for 4-bit math. It has to unpack NVFP4
#     back to a bigger format (BF16) before it can compute with it — in theory
#     that should erase any speed advantage from using the smaller format.
#   - We tested it anyway (2026-07-08, one request at a time): this image was
#     actually FASTER than the FP8 image — 65.6 tokens/sec vs FP8's 60.5. Best
#     guess why: NVFP4 still has to move half as much data through memory as
#     FP8 does, and on this workload that saving mattered more than the cost of
#     unpacking it. We haven't tested this with several requests running at
#     once (only one at a time so far), or checked whether NVFP4's answers are
#     as accurate as FP8's. See ../qwen-qwen3.6-35b-a3b-fp8-sglang for the FP8
#     image and its numbers.
#   - The other real benefit, regardless of speed: NVFP4 weights are ~20 GB vs
#     ~70 GB for the unquantized model — a much faster download, and it leaves
#     far more memory free for the KV cache (the model's working memory).
#   - The linear-attention layers are kept at a higher precision (8-bit) by
#     NVIDIA's quantization tool, to protect accuracy on the parts of the model
#     most sensitive to rounding error.
#   - Max 4 requests running at once (--max-running-requests).
#   - We tried a different attention kernel (flashinfer instead of the default
#     triton) and it made no difference here — 65.3 vs 65.6 tok/s — unlike the
#     FP8 image, where switching kernels gave a real ~30% speedup.
#
# MTP (a speed feature) — now available as of v0.5.15 (see EXTRA_ARGS default
# in docker-compose.yml). MTP lets the model predict several words at once
# instead of one at a time, which speeds up generation. Requires the
# triton-target-verify-mask-buffer.py patch below — without it, MTP combined
# with CUDA graphs crashes the scheduler once a batch's context usage crosses
# a threshold (see the patch file and RESEARCH_NOTES.md for details).
#
# Base image: CUDA 13.x is required for sm_121a, and Qwen3.6 (qwen3_5_moe arch)
# modeling support requires SGLang >= v0.5.13. v0.5.15 (2026-07-10) is the
# first release with native Qwen3.6 ModelOpt-mixed NVFP4 support, superseding
# the main-branch nightly this image previously pinned for that fix.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.15-cu130
FROM --platform=linux/arm64 ${SGLANG_IMAGE}

ENV MODEL_ID="nvidia/Qwen3.6-35B-A3B-NVFP4" \
    HOST="0.0.0.0" \
    PORT="30000" \
    QUANTIZATION="auto" \
    KV_CACHE_DTYPE="auto" \
    CONTEXT_LEN="262144" \
    MEM_FRACTION="0.85" \
    MAX_RUNNING_REQUESTS="4" \
    REASONING_PARSER="qwen3" \
    TOOL_CALL_PARSER="qwen3_coder" \
    ATTENTION_BACKEND="triton" \
    EXTRA_ARGS="" \
    HF_HOME="/root/.cache/huggingface" \
    # Point Triton at CUDA 13.0's ptxas instead of PyTorch's bundled one.
    # The bundled ptxas predates SM_121 and rejects --gpu-name=sm_121a, causing
    # JIT compilation failures for attention and other kernels at runtime.
    # /usr/local/cuda/bin/ptxas (from CUDA 13.0 in this image) knows SM_121a natively.
    TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas"

COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Generic SGLang bug (also present on the compressed-tensors sibling image):
# the CUDA-graph custom_mask buffer for MTP/EAGLE target-verify under-
# allocates by num_draft_tokens**2 * max_bs elements, crashing the scheduler
# once a batch's context usage crosses that margin — see the patch file for
# details.
COPY patches/triton-target-verify-mask-buffer.py /tmp/triton-target-verify-mask-buffer.py
RUN python3 /tmp/triton-target-verify-mask-buffer.py && rm /tmp/triton-target-verify-mask-buffer.py

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
