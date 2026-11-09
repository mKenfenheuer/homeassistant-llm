#!/bin/bash
# HF model (bf16) -> GGUF f16 -> Q4_K_M, using the official llama.cpp image.
# Usage: ./run_export.sh [hf_model_dir] [output_name]
#   default: output/merged -> model/qwen2.5-3b-ha-Q4_K_M.gguf
set -euo pipefail
cd "$(dirname "$0")"

SRC="${1:-output/merged}"
NAME="${2:-qwen2.5-3b-ha}"
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
mkdir -p model

SRC_ABS="$(realpath "$SRC")"

# transformers 5 writes tokenizer_config.json in a format the converter
# (transformers 4 in the image) cannot read. LoRA does not touch the
# tokenizer, so use the original files of the base model.
BASE_SNAP="$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/*/ | head -1)"
for f in tokenizer.json tokenizer_config.json vocab.json merges.txt; do
    cp -L "$BASE_SNAP/$f" "$SRC_ABS/$f"
done
rm -f "$SRC_ABS/chat_template.jinja"
docker run --rm --user "$(id -u):$(id -g)" \
    -v "$SRC_ABS":/src:ro -v "$PWD/model":/out "$IMG" \
    --convert /src --outtype f16 --outfile "/out/${NAME}-f16.gguf"

# Importance matrix from our own conversations (src/make_calib.py): without it
# Q4_K_M can break single training runs (v13: 291/300 in f16, 243/300 in Q4).
CALIB="${CALIB:-model/calib.txt}"
QARGS=()
if [ -f "$CALIB" ]; then
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/model":/out \
        --entrypoint /app/llama-imatrix "$IMG" -m "/out/${NAME}-f16.gguf" -f "/out/$(basename "$CALIB")" \
        -o "/out/${NAME}.imatrix" -ngl 99 -c 4096
    QARGS=(--imatrix "/out/${NAME}.imatrix")
fi
for Q in ${QUANTS:-Q4_K_M}; do
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/model":/out "$IMG" \
        --quantize "${QARGS[@]}" "/out/${NAME}-f16.gguf" "/out/${NAME}-${Q}.gguf" "$Q"
done

rm -f "model/${NAME}-f16.gguf"
ls -lh model/${NAME}-*.gguf
