#!/usr/bin/env python3
"""Fix an under-sized CUDA-graph custom_mask buffer for MTP/EAGLE target-verify.

This is a generic SGLang bug, unrelated to quantization format — it affects
any hybrid linear-attention (GDN/mamba) model using speculative decoding
(MTP/EAGLE "target_verify" mode) together with CUDA graphs on the triton
attention backend.

TritonAttnBackend.init_cuda_graph_state allocates the graph-replay custom_mask
buffer as:

    max_num_tokens * max_context_len

where max_num_tokens = max_bs * num_draft_tokens. But the real mask built in
eagle_info.py's generate_attn_arg_prefill (see its own
"# FIXME(attn): temporary fix for custom mask padding with cuda graph"
comment) sizes the mask as:

    paged_kernel_lens_sum * draft_token_num + draft_token_num**2 * batch_size

At the worst case (paged_kernel_lens_sum == bs * max_context_len, i.e. every
request in the batch near-full on context), the real mask needs
max_num_tokens * max_context_len + draft_token_num**2 * bs — exactly
draft_token_num**2 * bs MORE than the graph buffer allocates. The trailing
term is missing entirely from the buffer's sizing formula.

Symptom (reproduced 2026-07-11 on Qwen3.6-35B-A3B, num_draft_tokens=4,
max_bs=4, context_length=131072): triton_backend.py's
_update_target_verify_buffers does
`custom_mask[: spec_info.custom_mask.shape[0]] = spec_info.custom_mask`,
which crashes once spec_info.custom_mask outgrows the fixed buffer:

    RuntimeError: The expanded size of the tensor (2097152) must match the
    existing size (2097216) at non-singleton dimension 0.

2097216 - 2097152 = 64 = num_draft_tokens**2 * max_bs = 4**2 * 4, confirming
the missing term exactly. This crashes the scheduler process (the container
comes up "healthy" and serves fine at low concurrency, then dies the moment a
batch's total context usage crosses the threshold — non-deterministic from a
caller's perspective, since it depends on how full each request's context
happens to be, not just batch size).

Fix: pad the buffer allocation by the same trailing term the real mask uses,
at its worst case (using max_bs, since any real batch size b <= max_bs).
"""
import pathlib

path = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/layers/attention/triton_backend.py"
)
src = path.read_text()

old = """        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device=self.device,
            )"""

new = """        if not self.skip_prefill:
            # Patched in (see patches/triton-target-verify-mask-buffer.py):
            # pad by num_draft_tokens**2 * max_bs to match the worst-case size
            # of the real target-verify mask built in eagle_info.py, which
            # this buffer's original formula omitted entirely.
            num_draft_tokens_for_mask = self.num_draft_tokens or 1
            self.cuda_graph_custom_mask = torch.zeros(
                (
                    max_num_tokens * self.max_context_len
                    + num_draft_tokens_for_mask**2 * max_bs
                ),
                dtype=torch.uint8,
                device=self.device,
            )"""

if new in src:
    print("already patched")
elif old in src:
    path.write_text(src.replace(old, new))
    print(f"patched {path}")
else:
    raise SystemExit(
        f"ERROR: expected block not found in {path} — sglang source has "
        "changed; re-check whether this patch is still needed/correct."
    )
