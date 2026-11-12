#!/bin/bash
# Train the model of one language on a rented GPU (RunPod pod with a PyTorch
# image, >= 24 GB VRAM, ~40 GB disk). Run inside the repository checkout:
#   HA_LANG=en scripts/runpod_train.sh
# Result: output/<lang>/lora (adapter, ~240 MB) and output/<lang>/train.log.
# Copy the adapter back; export and evaluation run on the training server
# (run_export.sh needs Docker for llama.cpp).
set -euo pipefail
cd "$(dirname "$0")/.."
L="${HA_LANG:?set HA_LANG}"
python -c "import torch; assert torch.cuda.is_available(), 'no GPU'; print(torch.cuda.get_device_name(0))"
pip install -q -r requirements-train.txt
mkdir -p "output/$L"
# same data as on the training server: per_case and seed from run_train.sh
PY=python HA_LANG="$L" ./run_train.sh 2>&1 | tee "output/$L/train.log"
# the merged bf16 model is not needed remotely (6 GB); keep only the adapter
rm -rf "output/$L/merged"
ls -la "output/$L/lora"
