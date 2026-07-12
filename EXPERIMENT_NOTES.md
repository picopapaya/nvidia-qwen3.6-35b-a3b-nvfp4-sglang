## 2026-07-12

### Does SGLang v0.5.15 still need the ModelOpt mixed-precision routing patch this image carried?

**Findings**

No — rebuilding on `lmsysorg/sglang:v0.5.15-cu130` and running the existing patch script against it printed "already patched": the exact code block the patch would have inserted is already present in v0.5.15's own `model_config.py`. The fix landed upstream (the release notes list "Qwen3.6 ModelOpt mixed NVFP4" support). Removed the now-redundant `patches/modelopt-mixed-routing.py` and its Dockerfile step.

**Conclusion**

This image no longer carries any patches for its own quantization format — only the generic MTP/CUDA-graph patch below, shared with the compressed-tensors sibling.

---

### Does rebasing onto v0.5.15 change baseline decode speed?

**Findings**

Measured solo, full budget (`MEM_FRACTION=0.5`, `CONTEXT_LEN=131072`): 67.2 / 109.5 / 178.1 agg tok/s at concurrency 1/2/4 — matches (slightly exceeds) the pre-rebase nightly's 68.1/111.4/181.3 within noise.

**Conclusion**

Neutral rebase, same as observed on the compressed-tensors sibling.

---

### Is MTP possible on this image now, given it was previously confirmed unavailable for SGLang here (2026-07-08 entry below)?

**Findings**

Yes. That 2026-07-08 finding was accurate for the nightly available at the time; v0.5.15 changes this. `--speculative-algorithm NEXTN` works using the checkpoint's own unquantized MTP head (19 tensors, same as the Unsloth checkpoint's). Solo/full-budget/isolated numbers, patched (see the compressed-tensors sibling's 2026-07-12 entry for the full crash root-cause writeup — the bug and fix are identical here, since it lives in SGLang's generic triton attention backend, not this image's ModelOpt quantization code):

| Config | c1 agg tok/s | c2 agg tok/s | c4 agg tok/s |
|---|---|---|---|
| v0.5.15 baseline (no MTP) | 67.2 ± 0.5 | 109.5 ± 0.7 | 178.1 ± 0.5 |
| + MTP + fused-qk-norm-rope, CUDA graph **patched** | **94.9 ± 5.4** | **137.6 ± 19.1** | **210.5 ± 4.5** |

Without the `triton-target-verify-mask-buffer.py` patch, this crashes the same way the compressed-tensors sibling did (same generic SGLang bug, not quantization-specific).

**Conclusion**

MTP + fused-qk-norm-rope (patched) is the new default (`docker-compose.yml`'s `EXTRA_ARGS` fallback): +41%/+26%/+18% over baseline at concurrency 1/2/4, no coherence loss. This image's gain from MTP is larger than the Unsloth sibling's (+29%/+29%/+14%) — not yet investigated why.

---

### Does raising MEM_FRACTION to 0.65 (from the co-resident 0.38) and CONTEXT_LEN to 131072 (128K, from 65536) change decode speed?

**Findings**

Configuration: `MEM_FRACTION=0.65`, `CONTEXT_LEN=131072`, `MAX_RUNNING_REQUESTS=4` (fixed, not overridable), `EXTRA_ARGS` unchanged from the persisted default (`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 --enable-fused-qk-norm-rope --cuda-graph-max-bs 8`). Measured solo (the Unsloth sibling was stopped) — verified via the startup log, not just the healthcheck: `mem_fraction_static=0.65`, `context_length=131072`, `max_running_requests=4` (unclamped), KV pool `max_total_num_tokens=2422030`, coherent output.

| Concurrency | agg tok/s (mean ± stdev) | per-request tok/s (mean ± stdev) |
|---|---|---|
| 1 | 95.0 ± 5.1 | 95.1 ± 5.1 |
| 2 | 146.6 ± 5.3 | 75.1 ± 1.6 |
| 4 | 210.9 ± 5.0 | 54.2 ± 1.5 |

Statistically identical to the 2026-07-12 MTP entry above (`MEM_FRACTION=0.5`/`CONTEXT_LEN=131072`: 94.9/137.6/210.5), aside from lower run-to-run variance at concurrency 2 (this run: ±5.3; that one: ±19.1, likely a warm-up artifact in the earlier run rather than a real difference).

**Conclusion**

`MEM_FRACTION` and `CONTEXT_LEN` are capacity knobs (how much KV cache / how long a context fits), not speed knobs — raising them well past what's strictly needed doesn't move decode throughput once `max_running_requests` is already unclamped at 4. The box currently has enough free memory (with the Unsloth sibling stopped) to run this generous, non-co-resident configuration.

---

## 2026-07-08

### Is this image's NVFP4 quantization actually faster than the FP8 sibling on the GB10, given the chip has to unpack NVFP4 back to BF16 before it can compute with it?

**Findings**

Measured one request at a time on this box:

| Config | tokens/sec | vs FP8 (60.5 tok/s) |
|---|---|---|
| This image, default settings (`triton`) | 65.6 | 8% faster |
| This image, alternate kernel (`flashinfer`) | 65.3 | 8% faster (switching kernels made no real difference) |

This contradicts an earlier note in the FP8 image's docs claiming FP8 was faster — that note was based on other people's online benchmarks, not testing on this machine, and didn't hold up (corrected in `../qwen-qwen3.6-35b-a3b-fp8-sglang/README.md`). Switching to the `flashinfer` kernel made no difference for this image — unlike the FP8 image, where it was worth a real ~30% speedup. The slow part here is likely the "unpacking NVFP4" step, not the attention math itself.

**Conclusion**

NVFP4 wins on speed on this box. Not yet checked: several requests running at once (we only tested one at a time so far, and that's not how this box is actually used day to day), and whether NVFP4's answers are as accurate as FP8's (going down to a smaller number format can lose some precision, and we haven't measured how much that matters here).

---

### Can MTP (speculative decoding) be enabled on this image the way it can on the FP8 sibling?

**Findings**

NVIDIA's own page for this exact model says MTP works with it — but only through a different serving engine (vLLM), not the one used here (SGLang). SGLang's own documentation currently says NVFP4 is only officially supported for a smaller version of this model (27B, not the 35B version here) — this image is already off the beaten path just to run at all (that's why it needs a special nightly build with patches — see the README's "SGLang compatibility" section).

**Conclusion**

Unclear whether MTP is even possible with the current setup; nobody has confirmed it one way or the other for SGLang specifically. Worth checking again if a newer SGLang version comes out.

---

### Can this image run side by side with another ~30B-class model (e.g. the FP8 sibling) on a shared GPU?

**Findings**

The packaged defaults (`CONTEXT_LEN=262144`, `MEM_FRACTION=0.85`) assume this is the only large model on the GPU. Setting `CONTEXT_LEN=131072` and `MEM_FRACTION=0.5` in `.env` gives a roughly 50/50 memory split. Verified: both this image and the FP8 sibling started and ran healthy at the same time with these settings, and 4 concurrent requests at realistic prompt sizes (~7-8K tokens) all completed cleanly with no errors.

This image handles the split better than the FP8 sibling: its smaller weights (~22 GB) leave a much bigger KV cache pool at these settings — 277,600 tokens, vs FP8's 102,798 — enough for roughly two fully-independent 128K-context requests (FP8's pool can't even hold one). `MAX_RUNNING_REQUESTS` stays at the full 4 here, no auto-reduction needed.

**Conclusion**

Still short of 3-4 truly independent max-context requests, but comfortably handles 4 concurrent requests at realistic prompt sizes like the ones tested above. Treat `CONTEXT_LEN` in a shared-GPU setup as a safety ceiling for your longest realistic prompt, not a guarantee of full-context concurrency.
