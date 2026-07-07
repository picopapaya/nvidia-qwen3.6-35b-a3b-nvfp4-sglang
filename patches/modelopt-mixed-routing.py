#!/usr/bin/env python3
"""Backport of sgl-project/sglang main-branch ModelOpt MIXED_PRECISION routing.

v0.5.14 sends every ModelOpt MIXED_PRECISION checkpoint (except NemotronH) to
the DeepSeek-oriented w4afp8 path, which hardcodes 128x128 FP8 weight blocks
and crashes on Qwen3.6's narrow linear-attention projections
(output_partition_size=32 not divisible by block_n=128). Upstream main instead
routes checkpoints that contain NVFP4/W4A16_NVFP4 layers to modelopt_mixed,
whose per-layer handling loads them correctly. Apply that detection here.
"""
import pathlib

path = pathlib.Path(
    "/sgl-workspace/sglang/python/sglang/srt/configs/model_config.py"
)
src = path.read_text()

old = '''        if quant_algo == "MIXED_PRECISION":
            architectures = getattr(self.hf_config, "architectures", []) or []
            if getattr(self.hf_config, "model_type", None) == "nemotron_h" or any(
                arch.startswith("NemotronH") for arch in architectures
            ):
                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}
            return {"quant_method": "w4afp8", "quant_algo": quant_algo}'''

new = '''        if quant_algo == "MIXED_PRECISION":
            quantized_layers = json_quant_configs.get("quantized_layers") or {}
            has_modelopt_nvfp4_layers = any(
                str(layer_info.get("quant_algo", "")).upper()
                in ("NVFP4", "W4A16_NVFP4")
                for layer_info in quantized_layers.values()
                if isinstance(layer_info, dict)
            )
            if has_modelopt_nvfp4_layers:
                return {"quant_method": "modelopt_mixed", "quant_algo": quant_algo}
            return {"quant_method": "w4afp8", "quant_algo": quant_algo}'''

if new in src:
    print("already patched")
elif old in src:
    path.write_text(src.replace(old, new))
    print("patched: MIXED_PRECISION + NVFP4 layers -> modelopt_mixed")
else:
    raise SystemExit("ERROR: expected MIXED_PRECISION block not found; sglang version changed?")
