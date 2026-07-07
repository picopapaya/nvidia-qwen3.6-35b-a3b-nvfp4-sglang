#!/usr/bin/env bash
# Launch Qwen3.6-35B-A3B (NVFP4 pre-quantized) via SGLang on the GB10.
set -euo pipefail

echo "==> Qwen3.6-35B-A3B (NVFP4 pre-quantized) + SGLang on NVIDIA GB10 (DGX Spark)"
echo "    model=${MODEL_ID}  quant=${QUANTIZATION}  max-concurrent=${MAX_RUNNING_REQUESTS}"

# The model is not gated (Apache-2.0), so HF_TOKEN is optional — set it to
# avoid anonymous rate limits on the ~20 GB weight download.
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
else
  echo "    (HF_TOKEN not set — downloading anonymously)"
fi

# Optional prefetch: downloads weights into the mounted HF_HOME volume so the
# download is a visible, cacheable step separate from server startup.
if [[ "${PREFETCH:-1}" == "1" ]]; then
  echo "==> Downloading ${MODEL_ID} into ${HF_HOME} (cached on the mounted volume)"
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL_ID}')" || \
    echo "   (prefetch skipped/failed; SGLang will download on startup)"
fi

echo "==> Launching SGLang server on ${HOST}:${PORT}"
exec python3 -m sglang.launch_server \
  --model-path "${MODEL_ID}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --quantization "${QUANTIZATION}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --context-length "${CONTEXT_LEN}" \
  --mem-fraction-static "${MEM_FRACTION}" \
  --tp-size 1 \
  --max-running-requests "${MAX_RUNNING_REQUESTS}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --attention-backend "${ATTENTION_BACKEND}" \
  ${EXTRA_ARGS}
