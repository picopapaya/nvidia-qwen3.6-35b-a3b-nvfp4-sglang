# NVIDIA Qwen3.6-35B-A3B pre-quantized to NVFP4 (ModelOpt) served by SGLang, on the NVIDIA GB10 (DGX Spark).
#
# Model architecture — Qwen3.6-35B-A3B is a hybrid-attention Mixture-of-Experts VLM:
#   - 35B total parameters, ~3B active per token ("35B-A3B" = 35B total, 3B active)
#   - 256 experts per MoE layer; router activates 8 per token (+ shared expert)
#   - Hybrid: 3 of every 4 layers use linear attention (Gated DeltaNet), 1 uses full attention
#   - Multimodal: includes a vision encoder (text + image input)
#   - 262144-token native context (rope_theta 1e7)
#
# Quantization — NVFP4 on SM121a (GB10):
#   - No native FP4 GEMM kernel on SM12x; SGLang falls back to Marlin, which
#     dequantizes FP4 → BF16 inside the kernel, so the FP4 FLOPS advantage is lost.
#   - The win here is footprint: NVFP4 weights are ~20 GB vs ~70 GB BF16, halving
#     download time vs FP8 and leaving far more unified memory for KV cache.
#   - Linear-attention projections are kept at 8-bit by ModelOpt (mixed precision).
#   - Max concurrency capped at 4 via --max-running-requests.
#
# Base image: CUDA 13.x is required for sm_121a, and Qwen3.6 (qwen3_5_moe arch)
# modeling support requires SGLang >= v0.5.13.
ARG SGLANG_IMAGE=lmsysorg/sglang:v0.5.14-cu130
FROM --platform=linux/arm64 ${SGLANG_IMAGE}

ENV MODEL_ID="nvidia/Qwen3.6-35B-A3B-NVFP4" \
    HOST="0.0.0.0" \
    PORT="30000" \
    QUANTIZATION="modelopt_fp4" \
    KV_CACHE_DTYPE="auto" \
    CONTEXT_LEN="262144" \
    MEM_FRACTION="0.85" \
    MAX_RUNNING_REQUESTS="4" \
    REASONING_PARSER="qwen3" \
    TOOL_CALL_PARSER="qwen" \
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

EXPOSE 30000

HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/health" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
