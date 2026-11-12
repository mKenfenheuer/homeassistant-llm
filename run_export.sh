#!/bin/bash
# HF model (bf16) -> GGUF f16 -> Q4_K_M, using the official llama.cpp image.
# Usage: HA_LANG=de ./run_export.sh [hf_model_dir] [output_name]
#   default: output/<lang>/merged -> model/<lang>/qwen2.5-3b-ha-Q4_K_M.gguf
# The chat template templates/chat_template.<lang>.jinja carries the default
# system prompt of the language.
set -euo pipefail
cd "$(dirname "$0")"

L="${HA_LANG:-de}"
SRC="${1:-output/$L/merged}"
NAME="${2:-qwen2.5-3b-ha}"
TEMPLATE="templates/chat_template.$L.jinja"
[ -f "$TEMPLATE" ] || { echo "missing $TEMPLATE" >&2; exit 1; }
LANG_NAME="$(echo "$L" | tr '[:lower:]' '[:upper:]')"
IMG=ghcr.io/ggml-org/llama.cpp:full-cuda
mkdir -p "model/$L"

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
    -v "$SRC_ABS":/src:ro -v "$PWD/model/$L":/out "$IMG" \
    --convert /src --outtype f16 --outfile "/out/${NAME}-f16.gguf"

# Importance matrix from our own conversations (src/make_calib.py): without it
# Q4_K_M can break single training runs (v13: 291/300 in f16, 243/300 in Q4).
CALIB="${CALIB:-model/$L/calib.txt}"
QARGS=()
if [ -f "$CALIB" ]; then
    docker run --rm --gpus all --user "$(id -u):$(id -g)" -v "$PWD/model/$L":/out \
        --entrypoint /app/llama-imatrix "$IMG" -m "/out/${NAME}-f16.gguf" -f "/out/$(basename "$CALIB")" \
        -o "/out/${NAME}.imatrix" -ngl 99 -c 4096
    QARGS=(--imatrix "/out/${NAME}.imatrix")
fi
for Q in ${QUANTS:-Q4_K_M}; do
    docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/model/$L":/out "$IMG" \
        --quantize "${QARGS[@]}" "/out/${NAME}-f16.gguf" "/out/${NAME}-${Q}.gguf" "$Q"
    # chat template with the default system prompt of the language; renders
    # the same text as the Qwen2.5 template whenever a system message is set
    .venv/bin/gguf-new-metadata --force --chat-template-file "$TEMPLATE" \
        --general-name "Qwen2.5 3B Home Assistant ${LANG_NAME} ${NAME}" \
        "model/$L/${NAME}-${Q}.gguf" "model/$L/${NAME}-${Q}.tmpl.gguf" >/dev/null
    mv "model/$L/${NAME}-${Q}.tmpl.gguf" "model/$L/${NAME}-${Q}.gguf"
done

rm -f "model/$L/${NAME}-f16.gguf"
ls -lh model/$L/${NAME}-*.gguf
