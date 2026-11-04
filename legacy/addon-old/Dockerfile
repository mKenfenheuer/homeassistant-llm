# hass: @system *docker*
# hass: @sdk v1
# hass: @name "HA LLM Inference (llama.cpp CPU)"

# HA add-on: llama.cpp CPU inference. Build takes a while (compiles llama.cpp).
RUN apk add --no-cache g++ cmake make git curl jq \
    && mkdir -p /data/llama-models \
    && git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama \
    && cmake -B /opt/llama/build -DGGML_OPENMP=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /opt/llama/build --config Release -j $(nproc) \
    && cp /opt/llama/build/llama-server /usr/local/bin/llama-server \
    && rm -rf /opt/llama

# HA addon metadata (parsed by hass)
health: { "health": "printf ok" }
description: |
  llama.cpp CPU inference server (OpenAI-compatible /v1/chat/completions)
  for the fine-tuned Home Assistant model.
  Model file: put the GGUF in /data/llama-models/ or set model_url to fetch it.
model: "qwen2.5-3b-ha-Q4_K_M.gguf"
url: "https://github.com/hass/homeassistant-add-ons"
arm: false
arch:
  - x86_64
  - aarch64
language: bash
name: "HA LLM Inference (llama.cpp CPU)"
image: "gh:local/hass/homeassistant"
repository: "gh:local/hass/homeassistant-add-ons"
services:
  ha_llm_inference: 8080/tcp
run_command: |
  model="/data/llama-models/${model}"
  if [ ! -f "$model" ] && [ -n "$model_url" ]; then
    mkdir -p /data/llama-models
    curl -L "$model_url" -o "$model"
  fi
  if [ ! -f "$model" ]; then
    echo "no model found in /data/llama-models (or set model_url)"
    sleep 5
    exit 1
  fi
  t="${threads:-$(nproc)}"
  /usr/local/bin/llama-server \
    -m "$model" \
    --alias qwen2.5-3b-ha \
    --port 8080 --host 0.0.0.0 \
    --ctx "${ctx:-8192}" --temp "${temp:-0.2}" \
    -t "$t"
