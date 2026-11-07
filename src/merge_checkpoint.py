"""Merge one LoRA training checkpoint into the base model for export.

Usage: python src/merge_checkpoint.py output/lora/checkpoint-936 output/merged-c936
Then:  ./run_export.sh output/merged-c936 qwen2.5-3b-ha-v9-c936

Used for checkpoint selection: the acceptance tests decide which checkpoint
of a training run is released, not simply the last one.
"""
from __future__ import annotations

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, ckpt).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
    print("merged", ckpt, "->", out)


if __name__ == "__main__":
    main()
