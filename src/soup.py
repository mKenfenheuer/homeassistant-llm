"""Model soup: average the weight deltas of several LoRA runs on the same base.

Usage: python src/soup.py OUTDIR output/lora-v13 output/lora-v14 [...]

merged = base + mean_i(B_i @ A_i * alpha_i / r_i). Every run makes its own
small mistakes (v13: balcony irrigation, v14: dryer -> vacuum); averaging the
deltas keeps what the runs agree on and damps the idiosyncratic errors.
The deltas come straight from the LoRA matrices, one tensor at a time, so
RAM stays at one bf16 model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"


def main():
    out, runs = sys.argv[1], sys.argv[2:]
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    params = dict(model.named_parameters())
    files = []
    for r in runs:
        cfg = json.loads((Path(r) / "adapter_config.json").read_text())
        assert not cfg.get("use_dora") and not cfg.get("use_rslora")
        files.append((safe_open(str(Path(r) / "adapter_model.safetensors"), "pt"),
                      cfg["lora_alpha"] / cfg["r"]))
    keys = [k for k in files[0][0].keys() if k.endswith("lora_A.weight")]
    with torch.no_grad():
        for ka in keys:
            kb = ka.replace("lora_A", "lora_B")
            name = ka.removeprefix("base_model.model.").replace(".lora_A.weight", ".weight")
            delta = sum(f.get_tensor(kb).float() @ f.get_tensor(ka).float() * s
                        for f, s in files) / len(files)
            p = params[name]
            p.copy_((p.float() + delta).to(p.dtype))
    print(len(keys), "tensors averaged over", len(runs), "runs")
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)


if __name__ == "__main__":
    main()
