#!/bin/bash
# Start whisper-server with the options from the app configuration. The model
# comes from the whisper.cpp repository on Hugging Face and is checked with the
# SHA-256 Hugging Face reports. whisper-server uses the first GPU it finds
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

MODEL_DIR=/share/ha_whisper
HF_REPO=ggerganov/whisper.cpp
MODEL="$(opt model)"
MODEL="${MODEL:-auto}"
LANGUAGE="$(opt language)"
LANGUAGE="${LANGUAGE:-auto}"
USE_GPU="$(opt gpu)"
USE_GPU="${USE_GPU:-true}"
THREADS="$(opt threads)"
EXTRA="$(opt extra_args)"

# GPU: the same llama.cpp backends whisper-server loads
ON_GPU=false
if [ "$USE_GPU" = "true" ]; then
    DEVICE_LIST="$(/app/llama-server --list-devices 2>/dev/null | grep -E '^ +[A-Za-z]+[0-9]+:' || true)"
    for backend in CUDA Vulkan; do
        DEVICES="$(echo "$DEVICE_LIST" | grep -E "^ +${backend}[0-9]+:" || true)"
        if [ -n "$DEVICES" ]; then
            ON_GPU=true
            echo "[INFO] GPU (${backend}):"
            echo "$DEVICES" | head -1
            break
        fi
    done
fi
if [ "$ON_GPU" = false ]; then
    echo "[INFO] No GPU used, speech recognition runs on the CPU."
fi

# auto: the large turbo model on a GPU, the small model on the CPU
if [ "$MODEL" = "auto" ]; then
    if $ON_GPU; then MODEL=large-v3-turbo; else MODEL=small; fi
fi
case "$MODEL" in
    tiny|base|small) FILE="ggml-${MODEL}-q5_1.bin" ;;
    medium|large-v3|large-v3-turbo) FILE="ggml-${MODEL}-q5_0.bin" ;;
    *) fail "Unknown model ${MODEL}." ;;
esac
MODEL_PATH="${MODEL_DIR}/${FILE}"

# SHA-256 of the file as Hugging Face reports it
remote_sha() {
    curl -fsL --max-time 30 "https://huggingface.co/api/models/${HF_REPO}/tree/main" 2>/dev/null \
        | python3 -c '
import json, sys
for f in json.load(sys.stdin):
    if f.get("path") == sys.argv[1]:
        print((f.get("lfs") or {}).get("oid", ""))
' "$FILE"
}

if [ ! -f "$MODEL_PATH" ]; then
    SHA="$(remote_sha || true)"
    [ -n "$SHA" ] || fail "Can't get the checksum of ${FILE} from Hugging Face. A restart tries again."
    mkdir -p "$MODEL_DIR"
    echo "[INFO] Downloading model ${FILE} to ${MODEL_DIR} (an interrupted download resumes)"
    if ! curl -fL --retry 5 --retry-delay 10 --no-progress-meter -C - \
            -o "${MODEL_PATH}.part" "https://huggingface.co/${HF_REPO}/resolve/main/${FILE}"; then
        echo "${SHA}  ${MODEL_PATH}.part" | sha256sum -c --status - \
            || fail "Download failed. A restart tries again."
    fi
    if ! echo "${SHA}  ${MODEL_PATH}.part" | sha256sum -c --status -; then
        rm -f "${MODEL_PATH}.part"
        fail "Checksum mismatch, file deleted. A restart tries again."
    fi
    mv "${MODEL_PATH}.part" "$MODEL_PATH"
    echo "[INFO] Download complete"
fi

# Remove other models this app downloaded
for f in "$MODEL_DIR"/ggml-*.bin "$MODEL_DIR"/ggml-*.bin.part; do
    if [ -f "$f" ] && [ "$f" != "$MODEL_PATH" ]; then
        echo "[INFO] Removing old model ${f}"
        rm -f "$f"
    fi
done

SERVER=""
trap '[ -n "$SERVER" ] && kill "$SERVER" 2>/dev/null; wait; exit 0' TERM INT
# 0 as soon as whisper-server answers, 1 if it exits before
wait_ready() {
    while kill -0 "$SERVER" 2>/dev/null; do
        curl -sf -o /dev/null http://127.0.0.1:8080/ && return 0
        sleep 1
    done
    wait "$SERVER" 2>/dev/null || true
    return 1
}

# If another app holds the GPU memory, loading fails: wait until it has
# finished loading and try again, at last on the CPU.
modes=(cpu)
$ON_GPU && modes=(gpu gpu gpu cpu)
attempt=0
STARTED=false
for mode in "${modes[@]}"; do
    attempt=$((attempt + 1))
    args=(-m "$MODEL_PATH" --host 0.0.0.0 --port 8080 -l "$LANGUAGE")
    if [ "${THREADS:-0}" != "0" ]; then
        args+=(-t "$THREADS")
    fi
    if [ "$mode" = cpu ]; then
        args+=(-ng)
    fi
    echo "[INFO] Starting whisper-server: ${args[*]} ${EXTRA}"
    # shellcheck disable=SC2086
    /app/whisper-server "${args[@]}" ${EXTRA} &
    SERVER=$!
    if wait_ready; then
        if [ "$mode" = cpu ] && $ON_GPU; then
            echo "[WARNING] Not enough GPU memory, running on the CPU. A restart tries the GPU again."
        fi
        STARTED=true
        break
    fi
    if [ "$attempt" -lt "${#modes[@]}" ]; then
        echo "[WARNING] whisper-server didn't start (attempt ${attempt} of ${#modes[@]}), trying again in $((attempt * 15)) s"
        sleep $((attempt * 15)) &
        wait $! || true
    fi
done
$STARTED || fail "whisper-server didn't start."

# Warm up with one second of silence, so the first request of a user doesn't
# wait for model buffers and GPU code
python3 -c 'import wave,sys; w=wave.open(sys.argv[1],"wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000); w.writeframes(bytes(32000)); w.close()' /tmp/warmup.wav
START=$(date +%s)
if curl -sf -o /dev/null --max-time 300 http://127.0.0.1:8080/inference -F file=@/tmp/warmup.wav; then
    echo "[INFO] Ready ($(( $(date +%s) - START )) s warm-up)"
fi
rm -f /tmp/warmup.wav

wait "$SERVER"
exit $?
