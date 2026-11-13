#!/bin/bash
# Dataset + LoRA SFT for one language. Needs a free CUDA GPU with >= 20 GB
# (RTX 3090). The venv must contain torch, transformers, peft, accelerate,
# datasets.
# Usage: HA_LANG=de ./run_train.sh [train_sft.py options]
#   data:   data/<lang>/train.jsonl, data/<lang>/eval.jsonl
#   output: output/<lang>/lora (adapter), output/<lang>/merged (bf16)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PY:-.venv/bin/python}"
L="${HA_LANG:-de}"
# samples per case: every case (intent, question kind, error, turn type) gets
# the same share; v16 doubles the 118 per case of v15 (2 epochs instead of 2.5)
PER_CASE="${PER_CASE:-236}"
SEED="${SEED:-18}"

$PY src/gen_dataset.py --lang "$L" "data/$L/train.jsonl" "per_case:$PER_CASE" "$SEED"
$PY src/gen_dataset.py --lang "$L" "data/$L/eval.jsonl" 300 "$((SEED + 1000))"   # held-out, other houses
$PY src/check_dataset.py --lang "$L" "data/$L/train.jsonl"   # answers must match the tool results

# continue from the adapter of the last model (INIT= for a fresh adapter);
# train_sft.py stops when the eval loss no longer drops
INIT="${INIT-output/$L/lora-v18}"
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    $PY src/train_sft.py --lang "$L" ${INIT:+--init-adapter "$INIT"} "$@"

echo "adapter: output/$L/lora, merged model: output/$L/merged -> next: HA_LANG=$L ./run_export.sh"
