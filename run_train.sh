#!/bin/bash
# Dataset + LoRA SFT. Needs a free CUDA GPU with >= 20 GB (RTX 3090).
# The venv must contain torch, transformers, peft, accelerate, datasets.
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python

$PY src/gen_dataset.py data/train.jsonl 7000 3     # training set (v3)
$PY src/gen_dataset.py data/eval.jsonl 300 1001    # held-out set, other houses
$PY src/check_dataset.py data/train.jsonl          # answers must match the tool results

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY src/train_sft.py --batch 2 --accum 8 "$@"

echo "adapter: output/lora, merged model: output/merged -> next: ./run_export.sh"
