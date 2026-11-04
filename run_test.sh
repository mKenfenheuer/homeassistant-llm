#!/bin/bash
# Evaluate a GGUF on data/eval.jsonl with llama-server on the GPU (Docker).
# Usage: ./run_test.sh [model/xyz.gguf] [number of samples]
set -euo pipefail
cd "$(dirname "$0")"
GGUF="${1:-model/qwen2.5-3b-ha-Q4_K_M.gguf}"
N="${2:-300}"
PORT=18081

docker rm -f ha-llm-eval >/dev/null 2>&1 || true
docker run -d --name ha-llm-eval --gpus all -p $PORT:8080 \
    -v "$(realpath "$(dirname "$GGUF")")":/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda --server \
    -m "/models/$(basename "$GGUF")" --host 0.0.0.0 --port 8080 \
    --jinja -ngl 99 -c 8192 -np 1 >/dev/null
until curl -sf localhost:$PORT/health >/dev/null; do sleep 2; done

.venv/bin/python -u src/eval_model.py http://127.0.0.1:$PORT data/eval.jsonl "$N"
docker rm -f ha-llm-eval >/dev/null
