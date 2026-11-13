#!/bin/bash
# Start llama-server with the options from the add-on configuration.
# Without model_url/model_path the add-on uses the official model. With
# auto_update (default) it follows latest.json in the Hugging Face repository:
# checked at start and every update_interval hours, a new model is downloaded
# while the server keeps running and then replaces the old one.
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

# Language of the model. All languages share the Hugging Face repository
# mkenfenheuer/qwen2.5-3b-homeassistant-GGUF; its models.json links every
# language to its model file (<language>/<file>.gguf).
# German keeps the model folder of earlier versions.
LANGUAGE="$(opt language)"
LANGUAGE="${LANGUAGE:-de}"
[[ "$LANGUAGE" =~ ^[a-z]{2,3}$ ]] || fail "Invalid language ${LANGUAGE}."
HF_REPO="${HA_LLM_REPO:-https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-GGUF/resolve/main}"
MANIFEST_URL="${HF_REPO}/models.json"
# prompt schema of the integration released with this add-on version; a model
# for a newer schema needs a newer integration and is not installed
SUPPORTED_SCHEMA=7
DEFAULT_URL=""
DEFAULT_SHA=""
LEGACY_URLS=()
if [ "$LANGUAGE" = "de" ]; then
    MODEL_DIR=/share/ha_llm
    # model that belongs to this add-on version (fallback without network)
    DEFAULT_URL="${HF_REPO}/de/qwen2.5-3b-ha-v18-Q4_K_M.gguf"
    DEFAULT_SHA="ff8d0a7468ffa30a5cfb2c40d80fc8096b84390068fe42e8279538bfdc90b3cb"
    # default URLs of earlier versions: treated like "empty", old file removed
    LEGACY_URLS=(
        "${HF_REPO}/qwen2.5-3b-ha-Q4_K_M.gguf"
        "${HF_REPO}/qwen2.5-3b-ha-v2-Q4_K_M.gguf"
        "${HF_REPO}/qwen2.5-3b-ha-v3-Q4_K_M.gguf"
        "${HF_REPO}/qwen2.5-3b-ha-v4-Q4_K_M.gguf"
        "${HF_REPO}/qwen2.5-3b-ha-v7-Q4_K_M.gguf"
        "${HF_REPO}/qwen2.5-3b-ha-v11-Q4_K_M.gguf"
        "${HF_REPO}/qwen2.5-3b-ha-v15-soup-Q4_K_M.gguf"
        "${HF_REPO}/qwen2.5-3b-ha-v15.1-Q4_K_M.gguf"
        "https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF/resolve/main/qwen2.5-3b-ha-v15.1-Q4_K_M.gguf"
        "${HF_REPO}/de/qwen2.5-3b-ha-v16-Q4_K_M.gguf"
        "${HF_REPO}/de/qwen2.5-3b-ha-v17-Q4_K_M.gguf"
    )
else
    MODEL_DIR="/share/ha_llm/${LANGUAGE}"
fi
# model chosen by the last automatic update (url and sha256, one per line)
STATE="${MODEL_DIR}/current_model"

MODEL_URL="$(opt model_url)"
MODEL_SHA="$(opt model_sha256)"
MODEL_PATH="$(opt model_path)"
HF_TOKEN="$(opt hf_token)"
CTX="$(opt ctx_size)"
THREADS="$(opt threads)"
EXTRA="$(opt extra_args)"
AUTO_UPDATE="$(opt auto_update)"
INTERVAL="$(opt update_interval)"
AUTO_UPDATE="${AUTO_UPDATE:-true}"
INTERVAL="${INTERVAL:-6}"
GPU_LAYERS="$(opt gpu_layers)"
GPU_LAYERS="${GPU_LAYERS:-99}"
GPU_MEMORY="$(opt gpu_memory)"
GPU_MEMORY="${GPU_MEMORY:-0}"

for u in "${LEGACY_URLS[@]+"${LEGACY_URLS[@]}"}"; do
    if [ "$MODEL_URL" = "$u" ] || [ "$MODEL_URL" = "$DEFAULT_URL" ]; then
        MODEL_URL=""
        MODEL_SHA=""
    fi
done
# official model: no own URL and no own file
MANAGED=false
if [ -z "$MODEL_URL" ] && [ -z "$MODEL_PATH" ]; then
    MANAGED=true
fi

# llama.cpp loads the backends it finds hardware for. With an NVIDIA GPU
# both CUDA and Vulkan list it, so take the devices of one backend only:
# CUDA, else Vulkan (AMD, Intel, ARM), else the CPU.
ON_GPU=false
GPU_DEVICES=""
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
if [ "$ON_GPU" = false ]; then
    echo "[INFO] No GPU found, the model runs on the CPU."
fi
# the AVX2 warning only applies to x86 CPUs
if [ "$ON_GPU" = false ] && [ "$(uname -m)" = x86_64 ] && ! grep -qw avx2 /proc/cpuinfo; then
    CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
    echo "[WARNING] ============================================================"
    echo "[WARNING] The CPU (${CPU_MODEL:-unknown}) does not support AVX2."
    echo "[WARNING] llama.cpp is then about 30 times slower: the first request"
    echo "[WARNING] takes about 20 minutes with 100 entities, every command 30-60 s."
    echo "[WARNING] Recommendation: run llama-server on a machine with AVX2 or a GPU."
    echo "[WARNING] ============================================================"
    if [ -n "${SUPERVISOR_TOKEN:-}" ]; then
        python3 - "$CPU_MODEL" <<'PY' || echo "[WARNING] Notification to Home Assistant failed"
import json, os, sys, urllib.request
cpu = sys.argv[1] or "unknown"
body = {
    "notification_id": "ha_llm_inference_no_avx2",
    "title": "HA LLM Inference: CPU without AVX2",
    "message": (
        f"The CPU of this host ({cpu}) does not support AVX2. llama.cpp is then "
        "about 30 times slower: the first request takes about 20 minutes with "
        "100 entities, every further command 30 to 60 seconds.\n\n"
        "Recommendation: run llama-server on a machine with AVX2 or a GPU and "
        "enter its URL in the HA LLM Agent integration."),
}
req = urllib.request.Request(
    "http://supervisor/core/api/services/persistent_notification/create",
    data=json.dumps(body).encode(), method="POST",
    headers={"Authorization": "Bearer " + os.environ["SUPERVISOR_TOKEN"],
             "Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=10).read()
PY
    fi
fi

# download URL SHA PATH: resumable, checks SHA-256, 0 on success
download() {
    local url="$1" sha="$2" path="$3" part="$3.part"
    local auth=()
    if [ -n "$HF_TOKEN" ]; then
        auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    mkdir -p "$(dirname "$path")"
    echo "[INFO] Downloading model ${url}"
    echo "[INFO] Target ${path} (about 1.9 GB, an interrupted download resumes)"
    if ! curl -fL --retry 5 --retry-delay 10 --no-progress-meter -C - \
            "${auth[@]}" -o "$part" "$url"; then
        # curl also fails with HTTP 416 if the .part file is already complete
        if [ -z "$sha" ] || ! echo "${sha}  ${part}" | sha256sum -c --status -; then
            echo "[WARNING] Download failed."
            return 1
        fi
    fi
    if [ -n "$sha" ]; then
        echo "[INFO] Checking SHA-256"
        if ! echo "${sha}  ${part}" | sha256sum -c --status -; then
            rm -f "$part"
            echo "[WARNING] Checksum mismatch, file deleted."
            return 1
        fi
    fi
    mv "$part" "$path"
    echo "[INFO] Download complete"
}

# prints "url sha256 version" of the newest model this add-on can run
manifest() {
    local auth=()
    if [ -n "$HF_TOKEN" ]; then
        auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    curl -fsL --max-time 30 "${auth[@]}" "$MANIFEST_URL" 2>/dev/null | python3 -c '
import json, re, sys
repo, supported, lang = sys.argv[1], int(sys.argv[2]), sys.argv[3]
m = json.load(sys.stdin).get("languages", {}).get(lang)
if m is None:
    print("[INFO] No model for language %s in models.json" % lang, file=sys.stderr)
    sys.exit(3)
f, sha = m["file"], m["sha256"].lower()
if not re.fullmatch(r"([a-z]{2,3}/)?[A-Za-z0-9._-]+\.gguf", f) or not re.fullmatch(r"[0-9a-f]{64}", sha):
    sys.exit("invalid manifest")
if int(m.get("schema", 0)) > supported:
    print("[INFO] Model %s needs schema %s, available after an app update"
          % (m.get("version"), m["schema"]), file=sys.stderr)
    sys.exit(2)
print(f"{repo}/{f}", sha, m.get("version", "?"))
' "$HF_REPO" "$SUPPORTED_SCHEMA" "$LANGUAGE"
}

# removes official model files other than the current one
cleanup() {
    local keep="$1" f
    local old=("${LEGACY_URLS[@]+"${LEGACY_URLS[@]}"}")
    [ -n "$DEFAULT_URL" ] && old+=("$DEFAULT_URL")
    if [ -f "$STATE.prev" ]; then
        old+=("$(head -1 "$STATE.prev")")
    fi
    for u in "${old[@]}"; do
        f="${MODEL_DIR}/$(basename "$u")"
        if [ "$f" != "$keep" ] && { [ -f "$f" ] || [ -f "$f.part" ]; }; then
            echo "[INFO] Removing old model ${f}"
            rm -f "$f" "$f.part"
        fi
    done
    rm -f "$STATE.prev"
}

set_state() {
    if [ -f "$STATE" ]; then
        cp "$STATE" "$STATE.prev"
    fi
    printf '%s\n%s\n' "$1" "$2" > "$STATE"
}

if $MANAGED; then
    NEW=""
    if [ "$AUTO_UPDATE" = "true" ]; then
        NEW="$(manifest)" || NEW=""
    fi
    if [ -n "$NEW" ]; then
        read -r MODEL_URL MODEL_SHA VERSION <<< "$NEW"
        echo "[INFO] Current model on Hugging Face: ${VERSION}"
    elif [ "$AUTO_UPDATE" = "true" ] && [ -f "$STATE" ] \
            && [ -f "${MODEL_DIR}/$(basename "$(head -1 "$STATE")")" ]; then
        echo "[INFO] Hugging Face not reachable, using the last downloaded model"
        MODEL_URL="$(sed -n 1p "$STATE")"
        MODEL_SHA="$(sed -n 2p "$STATE")"
    elif [ -n "$DEFAULT_URL" ]; then
        MODEL_URL="$DEFAULT_URL"
        MODEL_SHA="$DEFAULT_SHA"
    else
        fail "No model for language ${LANGUAGE} found on Hugging Face (${MANIFEST_URL})."
    fi
fi

if [ -z "$MODEL_PATH" ]; then
    [ -n "$MODEL_URL" ] || fail "model_url or model_path must be set."
    MODEL_PATH="${MODEL_DIR}/$(basename "${MODEL_URL%%\?*}")"
fi

if [ ! -f "$MODEL_PATH" ]; then
    [ -n "$MODEL_URL" ] || fail "Model file ${MODEL_PATH} is missing and model_url is empty."
    download "$MODEL_URL" "$MODEL_SHA" "$MODEL_PATH" || fail "No model available. A restart tries again."
fi
if $MANAGED; then
    if [ "$(head -1 "$STATE" 2>/dev/null)" != "$MODEL_URL" ]; then
        set_state "$MODEL_URL" "$MODEL_SHA"
    fi
    cleanup "$MODEL_PATH"
fi

SERVER=""
# Free memory of the first GPU in MiB, as llama.cpp reports it
gpu_free_mib() {
    /app/llama-server --list-devices 2>/dev/null \
        | grep -E "^ +${GPU_DEVICES%%,*}:" | sed -nE 's/.*, ([0-9]+) MiB free.*/\1/p' | head -1
}
# GPU arguments for a start attempt:
#   gpu  the configured layers, or with gpu_memory as many as fit the budget
#   fit  as many layers as fit the free memory (llama.cpp --fit)
#   cpu  no GPU
gpu_args() {
    local free
    case "$1" in
        gpu)
            if [ "${GPU_MEMORY}" != "0" ]; then
                free="$(gpu_free_mib)"
                free="${free:-0}"
                # --fit keeps the target free and fills the rest of the GPU, so
                # leave everything but the budget free. --fit doesn't count the
                # about 300 MiB of the CUDA or Vulkan context.
                local target=$(( free - GPU_MEMORY + 300 ))
                [ "$target" -lt 256 ] && target=256
                echo "[INFO] GPU memory budget ${GPU_MEMORY} MiB, ${free} MiB free" >&2
                echo "--device $GPU_DEVICES --fit on --fit-target $target"
            else
                echo "--device $GPU_DEVICES --n-gpu-layers $GPU_LAYERS"
            fi
            ;;
        fit) echo "--device $GPU_DEVICES --fit on" ;;
        cpu) echo "--device none" ;;
    esac
}
# 0 as soon as llama-server answers, 1 if it exits before
wait_ready() {
    while kill -0 "$SERVER" 2>/dev/null; do
        curl -sf -o /dev/null http://127.0.0.1:8080/health && return 0
        sleep 1
    done
    wait "$SERVER" 2>/dev/null || true
    SERVER=""
    return 1
}
# Starts llama-server. If another app holds the GPU memory, loading fails:
# wait until it has finished loading and try again, then with fewer layers on
# the GPU, at last on the CPU.
start_server() {
    local modes=(cpu) attempt=0 mode gpu
    if [ "$ON_GPU" = true ] && [ "${GPU_LAYERS}" != "0" ]; then
        modes=(gpu gpu gpu fit cpu)
        # with a budget, fewer layers mean the budget doesn't fit at all
        [ "${GPU_MEMORY}" != "0" ] && modes=(gpu gpu gpu cpu)
    fi
    for mode in "${modes[@]}"; do
        attempt=$((attempt + 1))
        local args=(-m "$MODEL_PATH" --host 0.0.0.0 --port 8080 --alias ha-llm
                    --ctx-size "$CTX" --parallel 1 --jinja --cache-reuse 256)
        if [ "${THREADS:-0}" != "0" ]; then
            args+=(--threads "$THREADS")
        fi
        gpu="$(gpu_args "$mode")"
        # shellcheck disable=SC2206
        args+=($gpu)
        echo "[INFO] Starting llama-server: ${args[*]} ${EXTRA}"
        # shellcheck disable=SC2086
        /app/llama-server "${args[@]}" ${EXTRA} &
        SERVER=$!
        if wait_ready; then
            [ "$mode" = gpu ] || echo "[WARNING] Not enough GPU memory, running with mode ${mode}. A restart tries the GPU again."
            return 0
        fi
        if [ "$attempt" -lt "${#modes[@]}" ]; then
            echo "[WARNING] llama-server didn't start (attempt ${attempt} of ${#modes[@]}), trying again in $((attempt * 15)) s"
            sleep $((attempt * 15)) &
            wait $! || true
        fi
    done
    fail "llama-server didn't start."
}
stop_server() {
    if [ -n "$SERVER" ]; then
        kill "$SERVER" 2>/dev/null || true
        wait "$SERVER" 2>/dev/null || true
        SERVER=""
    fi
}
trap 'stop_server; exit 0' TERM INT

if ! $MANAGED || [ "$AUTO_UPDATE" != "true" ]; then
    start_server
    wait "$SERVER" || fail "llama-server stopped."
    exit 0
fi

start_server
while true; do
    for _ in $(seq "${HA_LLM_CHECK_MINUTES:-$((INTERVAL * 60))}"); do
        sleep 60 &
        wait $! || true
        if ! kill -0 "$SERVER" 2>/dev/null; then
            wait "$SERVER" || true
            fail "llama-server stopped."
        fi
    done
    NEW="$(manifest)" || continue
    read -r NEW_URL NEW_SHA VERSION <<< "$NEW"
    if [ "$NEW_SHA" = "$MODEL_SHA" ]; then
        continue
    fi
    echo "[INFO] New model ${VERSION} available"
    NEW_PATH="${MODEL_DIR}/$(basename "$NEW_URL")"
    if [ "$NEW_PATH" = "$MODEL_PATH" ] || ! download "$NEW_URL" "$NEW_SHA" "$NEW_PATH"; then
        continue
    fi
    echo "[INFO] Switching to model ${VERSION}"
    stop_server
    MODEL_URL="$NEW_URL"; MODEL_SHA="$NEW_SHA"; MODEL_PATH="$NEW_PATH"
    set_state "$MODEL_URL" "$MODEL_SHA"
    cleanup "$MODEL_PATH"
    start_server
done
