#!/bin/bash
# Start llama-server with the options from the add-on configuration.
# Without model_url/model_path the add-on uses the official model. With
# auto_update (default) it follows latest.json in the Hugging Face repository:
# checked at start and every update_interval hours, a new model is downloaded
# while the server keeps running and then replaces the old one.
set -euo pipefail

OPTIONS=/data/options.json
opt() {
    python3 -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else str(v).lower() if isinstance(v,bool) else v)' "$OPTIONS" "$1"
}
fail() {
    echo "[ERROR] $*"
    sleep 30
    exit 1
}

MODEL_DIR=/share/ha_llm
HF_REPO="${HA_LLM_REPO:-https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF/resolve/main}"
MANIFEST_URL="${HF_REPO}/latest.json"
# prompt schema of the integration released with this add-on version; a model
# for a newer schema needs a newer integration and is not installed
SUPPORTED_SCHEMA=5
# model that belongs to this add-on version (fallback without network)
DEFAULT_URL="${HF_REPO}/qwen2.5-3b-ha-v15.1-Q4_K_M.gguf"
DEFAULT_SHA="1762d43726ece67157a105c96d332dcbc0227ed135435dcfa7b203b36431d6c7"
# default URLs of earlier versions: treated like "empty", old file removed
LEGACY_URLS=(
    "${HF_REPO}/qwen2.5-3b-ha-Q4_K_M.gguf"
    "${HF_REPO}/qwen2.5-3b-ha-v2-Q4_K_M.gguf"
    "${HF_REPO}/qwen2.5-3b-ha-v3-Q4_K_M.gguf"
    "${HF_REPO}/qwen2.5-3b-ha-v4-Q4_K_M.gguf"
    "${HF_REPO}/qwen2.5-3b-ha-v7-Q4_K_M.gguf"
    "${HF_REPO}/qwen2.5-3b-ha-v11-Q4_K_M.gguf"
    "${HF_REPO}/qwen2.5-3b-ha-v15-soup-Q4_K_M.gguf"
)
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
# cpu, cuda or vulkan, set by the Dockerfile of each add-on
BACKEND="${HA_LLM_BACKEND:-cpu}"
GPU_LAYERS="$(opt gpu_layers)"
GPU_LAYERS="${GPU_LAYERS:-99}"

for u in "${LEGACY_URLS[@]}"; do
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

ON_GPU=false
if [ "$BACKEND" != "cpu" ]; then
    # llama.cpp lists the GPUs of the backend; without one it runs on the CPU
    DEVICES="$(/app/llama-server --list-devices 2>&1 | grep -E '^ +[A-Za-z]+[0-9]+:' || true)"
    if [ -n "$DEVICES" ]; then
        ON_GPU=true
        echo "[INFO] GPU (${BACKEND}):"
        echo "$DEVICES"
    else
        echo "[INFO] Keine GPU für ${BACKEND} gefunden, das Modell läuft auf der CPU."
        echo "[INFO] Für die GPU braucht der Host den passenden Treiber, bei NVIDIA"
        echo "[INFO] die NVIDIA-Variante von Home Assistant OS."
    fi
fi
# the AVX2 warning only applies to x86 CPUs
if [ "$ON_GPU" = false ] && [ "$(uname -m)" = x86_64 ] && ! grep -qw avx2 /proc/cpuinfo; then
    CPU_MODEL="$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
    echo "[WARNING] ============================================================"
    echo "[WARNING] Die CPU (${CPU_MODEL:-unbekannt}) unterstützt kein AVX2."
    echo "[WARNING] llama.cpp rechnet dann etwa 30-mal langsamer: die erste Anfrage"
    echo "[WARNING] dauert bei 100 Entitäten rund 20 Minuten, jeder Befehl 30-60 s."
    echo "[WARNING] Empfehlung: llama-server auf einem Rechner mit AVX2 oder GPU."
    echo "[WARNING] ============================================================"
    if [ -n "${SUPERVISOR_TOKEN:-}" ]; then
        python3 - "$CPU_MODEL" <<'PY' || echo "[WARNING] Benachrichtigung an Home Assistant fehlgeschlagen"
import json, os, sys, urllib.request
cpu = sys.argv[1] or "unbekannt"
body = {
    "notification_id": "ha_llm_inference_no_avx2",
    "title": "HA LLM Inference: CPU ohne AVX2",
    "message": (
        f"Die CPU dieses Hosts ({cpu}) unterstützt kein AVX2. llama.cpp rechnet "
        "dann etwa 30-mal langsamer: Die erste Anfrage dauert bei 100 Entitäten "
        "rund 20 Minuten, jeder weitere Befehl 30 bis 60 Sekunden.\n\n"
        "Empfehlung: llama-server auf einem Rechner mit AVX2 oder GPU betreiben "
        "und dessen URL in der Integration HA LLM Agent eintragen."),
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
    echo "[INFO] Lade Modell ${url}"
    echo "[INFO] Ziel ${path} (ca. 1,9 GB, ein abgebrochener Download wird fortgesetzt)"
    if ! curl -fL --retry 5 --retry-delay 10 --no-progress-meter -C - \
            "${auth[@]}" -o "$part" "$url"; then
        # curl also fails with HTTP 416 if the .part file is already complete
        if [ -z "$sha" ] || ! echo "${sha}  ${part}" | sha256sum -c --status -; then
            echo "[WARNING] Download fehlgeschlagen."
            return 1
        fi
    fi
    if [ -n "$sha" ]; then
        echo "[INFO] Prüfe SHA-256"
        if ! echo "${sha}  ${part}" | sha256sum -c --status -; then
            rm -f "$part"
            echo "[WARNING] Prüfsumme stimmt nicht, Datei gelöscht."
            return 1
        fi
    fi
    mv "$part" "$path"
    echo "[INFO] Download abgeschlossen"
}

# prints "url sha256 version" of the newest model this add-on can run
manifest() {
    local auth=()
    if [ -n "$HF_TOKEN" ]; then
        auth=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    curl -fsL --max-time 30 "${auth[@]}" "$MANIFEST_URL" 2>/dev/null | python3 -c '
import json, re, sys
repo, supported = sys.argv[1], int(sys.argv[2])
m = json.load(sys.stdin)
f, sha = m["file"], m["sha256"].lower()
if not re.fullmatch(r"[A-Za-z0-9._-]+\.gguf", f) or not re.fullmatch(r"[0-9a-f]{64}", sha):
    sys.exit("ungültiges Manifest")
if int(m.get("schema", 0)) > supported:
    print("[INFO] Modell %s braucht Schema %s, erst nach einem Add-on-Update verfügbar"
          % (m.get("version"), m["schema"]), file=sys.stderr)
    sys.exit(2)
print(f"{repo}/{f}", sha, m.get("version", "?"))
' "$HF_REPO" "$SUPPORTED_SCHEMA"
}

# removes official model files other than the current one
cleanup() {
    local keep="$1" f
    local old=("${LEGACY_URLS[@]}" "$DEFAULT_URL")
    if [ -f "$STATE.prev" ]; then
        old+=("$(head -1 "$STATE.prev")")
    fi
    for u in "${old[@]}"; do
        f="${MODEL_DIR}/$(basename "$u")"
        if [ "$f" != "$keep" ] && { [ -f "$f" ] || [ -f "$f.part" ]; }; then
            echo "[INFO] Entferne altes Modell ${f}"
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
        echo "[INFO] Aktuelles Modell laut Hugging Face: ${VERSION}"
    elif [ "$AUTO_UPDATE" = "true" ] && [ -f "$STATE" ] \
            && [ -f "${MODEL_DIR}/$(basename "$(head -1 "$STATE")")" ]; then
        echo "[INFO] Hugging Face nicht erreichbar, nutze das zuletzt geladene Modell"
        MODEL_URL="$(sed -n 1p "$STATE")"
        MODEL_SHA="$(sed -n 2p "$STATE")"
    else
        MODEL_URL="$DEFAULT_URL"
        MODEL_SHA="$DEFAULT_SHA"
    fi
fi

if [ -z "$MODEL_PATH" ]; then
    [ -n "$MODEL_URL" ] || fail "model_url oder model_path muss gesetzt sein."
    MODEL_PATH="${MODEL_DIR}/$(basename "${MODEL_URL%%\?*}")"
fi

if [ ! -f "$MODEL_PATH" ]; then
    [ -n "$MODEL_URL" ] || fail "Modelldatei ${MODEL_PATH} fehlt und model_url ist leer."
    download "$MODEL_URL" "$MODEL_SHA" "$MODEL_PATH" || fail "Kein Modell verfügbar. Neustart versucht es erneut."
fi
if $MANAGED; then
    if [ "$(head -1 "$STATE" 2>/dev/null)" != "$MODEL_URL" ]; then
        set_state "$MODEL_URL" "$MODEL_SHA"
    fi
    cleanup "$MODEL_PATH"
fi

SERVER=""
start_server() {
    local args=(-m "$MODEL_PATH" --host 0.0.0.0 --port 8080 --alias ha-llm
                --ctx-size "$CTX" --parallel 1 --jinja --cache-reuse 256)
    if [ "${THREADS:-0}" != "0" ]; then
        args+=(--threads "$THREADS")
    fi
    if [ "$BACKEND" != "cpu" ]; then
        args+=(--n-gpu-layers "$GPU_LAYERS")
    fi
    echo "[INFO] Starte llama-server: ${args[*]} ${EXTRA}"
    # shellcheck disable=SC2086
    /app/llama-server "${args[@]}" ${EXTRA} &
    SERVER=$!
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
    wait "$SERVER"
    exit $?
fi

start_server
while true; do
    for _ in $(seq "${HA_LLM_CHECK_MINUTES:-$((INTERVAL * 60))}"); do
        sleep 60 &
        wait $! || true
        if ! kill -0 "$SERVER" 2>/dev/null; then
            wait "$SERVER" || true
            fail "llama-server wurde beendet."
        fi
    done
    NEW="$(manifest)" || continue
    read -r NEW_URL NEW_SHA VERSION <<< "$NEW"
    if [ "$NEW_SHA" = "$MODEL_SHA" ]; then
        continue
    fi
    echo "[INFO] Neues Modell ${VERSION} verfügbar"
    NEW_PATH="${MODEL_DIR}/$(basename "$NEW_URL")"
    if [ "$NEW_PATH" = "$MODEL_PATH" ] || ! download "$NEW_URL" "$NEW_SHA" "$NEW_PATH"; then
        continue
    fi
    echo "[INFO] Wechsle auf Modell ${VERSION}"
    stop_server
    MODEL_URL="$NEW_URL"; MODEL_SHA="$NEW_SHA"; MODEL_PATH="$NEW_PATH"
    set_state "$MODEL_URL" "$MODEL_SHA"
    cleanup "$MODEL_PATH"
    start_server
done
