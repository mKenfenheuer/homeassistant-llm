#!/bin/bash
# Start llama-server with a text-to-speech model (POST /tts). The model files
# come from the ggml-org repositories on Hugging Face and are checked with the
# SHA-256 Hugging Face reports. The server uses the GPU devices of one backend
# (CUDA before Vulkan), otherwise the CPU.
set -euo pipefail

OPTIONS=/data/options.json
# The CUDA driver compiles GPU code that isn't prebuilt for the card (e.g.
# Pascal) on first use. Keep its cache across restarts.
export CUDA_CACHE_PATH=/data/cuda-cache
export CUDA_CACHE_MAXSIZE=2147483648
opt() {
    python3 -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else str(v).lower() if isinstance(v,bool) else v)' "$OPTIONS" "$1"
}
fail() {
    echo "[ERROR] $*"
    sleep 30
    exit 1
}

MODEL_DIR=/share/ha_tts
MODEL="$(opt model)"
MODEL="${MODEL:-lfm3-omni}"
USE_GPU="$(opt gpu)"
USE_GPU="${USE_GPU:-true}"
THREADS="$(opt threads)"
EXTRA="$(opt extra_args)"

case "$MODEL" in
    lfm3-omni)
        REPO=ggml-org/LFM3-Omni-2B-GGUF
        FILE=LFM3-Omni-2B-Q4_K_M.gguf
        MMPROJ=mmproj-LFM3-Omni-2B-Q8_0.gguf
        ;;
    qwen3-tts)
        REPO=ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF
        FILE=Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf
        MMPROJ=mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf
        ;;
    *) fail "Unknown model ${MODEL}." ;;
esac

# Reference recordings for voice cloning (Qwen3-TTS), used by the integration
mkdir -p "${MODEL_DIR}/voices"

# GPU devices of one backend: an NVIDIA GPU is listed by CUDA and Vulkan
ON_GPU=false
GPU_DEVICES=""
if [ "$USE_GPU" = "true" ]; then
    DEVICE_LIST="$(/app/llama-server --list-devices 2>/dev/null | grep -E '^ +[A-Za-z]+[0-9]+:' || true)"
    for backend in CUDA Vulkan; do
        DEVICES="$(echo "$DEVICE_LIST" | grep -E "^ +${backend}[0-9]+:" || true)"
        if [ -n "$DEVICES" ]; then
            ON_GPU=true
            GPU_DEVICES="$(echo "$DEVICES" | sed -E 's/^ +([A-Za-z]+[0-9]+):.*/\1/' | paste -sd, -)"
            echo "[INFO] GPU (${backend}):"
            echo "$DEVICES"
            break
        fi
    done
fi
if [ "$ON_GPU" = false ]; then
    echo "[INFO] No GPU used, speech synthesis runs on the CPU."
fi

# SHA-256 of a file as Hugging Face reports it
remote_sha() {
    curl -fsL --max-time 30 "https://huggingface.co/api/models/${REPO}/tree/main" 2>/dev/null \
        | python3 -c '
import json, sys
for f in json.load(sys.stdin):
    if f.get("path") == sys.argv[1]:
        print((f.get("lfs") or {}).get("oid", ""))
' "$1"
}

download() {
    local file="$1" path="${MODEL_DIR}/$1" sha
    [ -f "$path" ] && return 0
    sha="$(remote_sha "$file" || true)"
    [ -n "$sha" ] || fail "Can't get the checksum of ${file} from Hugging Face. A restart tries again."
    echo "[INFO] Downloading ${file} to ${MODEL_DIR} (an interrupted download resumes)"
    if ! curl -fL --retry 5 --retry-delay 10 --no-progress-meter -C - \
            -o "${path}.part" "https://huggingface.co/${REPO}/resolve/main/${file}"; then
        echo "${sha}  ${path}.part" | sha256sum -c --status - \
            || fail "Download failed. A restart tries again."
    fi
    if ! echo "${sha}  ${path}.part" | sha256sum -c --status -; then
        rm -f "${path}.part"
        fail "Checksum mismatch, file deleted. A restart tries again."
    fi
    mv "${path}.part" "$path"
    echo "[INFO] Download complete"
}

download "$FILE"
download "$MMPROJ"

# Remove model files of the other model
for f in "$MODEL_DIR"/*.gguf "$MODEL_DIR"/*.gguf.part; do
    if [ -f "$f" ] && [ "$f" != "${MODEL_DIR}/${FILE}" ] && [ "$f" != "${MODEL_DIR}/${MMPROJ}" ]; then
        echo "[INFO] Removing old model ${f}"
        rm -f "$f"
    fi
done

args=(-m "${MODEL_DIR}/${FILE}" -mm "${MODEL_DIR}/${MMPROJ}" --host 0.0.0.0 --port 8080
      --alias "$MODEL" --parallel 1)
if [ "${THREADS:-0}" != "0" ]; then
    args+=(--threads "$THREADS")
fi
if [ "$ON_GPU" = true ]; then
    args+=(--device "$GPU_DEVICES" --n-gpu-layers 99)
else
    args+=(--device none)
fi
echo "[INFO] Starting llama-server: ${args[*]} ${EXTRA}"
# shellcheck disable=SC2086
/app/llama-server "${args[@]}" ${EXTRA} &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null; wait "$SERVER"; exit 0' TERM INT

# Warm up with a short sentence, so the first answer doesn't wait for GPU code
for _ in $(seq 180); do
    curl -sf -o /dev/null http://127.0.0.1:8080/health && break
    kill -0 "$SERVER" 2>/dev/null || break
    sleep 1
done
START=$(date +%s)
if curl -sf -o /dev/null --max-time 300 http://127.0.0.1:8080/tts \
        -H 'Content-Type: application/json' -d '{"input": "Hello.", "lang": "en"}'; then
    echo "[INFO] Ready ($(( $(date +%s) - START )) s warm-up)"
fi

wait "$SERVER"
