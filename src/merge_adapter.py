"""Merge a LoRA adapter (e.g. a training checkpoint) into the base model.

Runs on CPU, so it works while the GPU is still training.
Usage: python src/merge_adapter.py output/lora/checkpoint-375 output/merged-375
"""

import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen2.5-3B-Instruct"
adapter, out = sys.argv[1], sys.argv[2]
model = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.bfloat16, device_map={"": "cpu"})
model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
model.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(BASE).save_pretrained(out)
print("merged ->", out)
