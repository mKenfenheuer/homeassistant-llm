#!/bin/bash
# Start llama-server with the options from the add-on configuration.
# Downloads the model from model_url (Hugging Face) on first start.
set -euo pipefail

OPTIONS=/data/options.json
opt() {
    python3 -c 'import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print("" if v is None else v)' "$OPTIONS" "$1"
}
fail() {
    echo "[ERROR] $*"
    sleep 30
    exit 1
}

MODEL_URL="$(opt model_url)"
MODEL_SHA="$(opt model_sha256)"
MODEL_PATH="$(opt model_path)"
HF_TOKEN="$(opt hf_token)"
CTX="$(opt ctx_size)"
THREADS="$(opt threads)"
EXTRA="$(opt extra_args)"

# --- CPU check: without AVX2 llama.cpp falls back to slow generic kernels
if ! grep -qw avx2 /proc/cpuinfo; then
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

# default location: /share/ha_llm/<file name from the URL>
if [ -z "$MODEL_PATH" ]; then
    [ -n "$MODEL_URL" ] || fail "model_url oder model_path muss gesetzt sein."
    MODEL_PATH="/share/ha_llm/$(basename "${MODEL_URL%%\?*}")"
fi

if [ ! -f "$MODEL_PATH" ]; then
    [ -n "$MODEL_URL" ] || fail "Modelldatei ${MODEL_PATH} fehlt und model_url ist leer."
    mkdir -p "$(dirname "$MODEL_PATH")"
    PART="${MODEL_PATH}.part"
    AUTH=()
    if [ -n "$HF_TOKEN" ]; then
        AUTH=(-H "Authorization: Bearer ${HF_TOKEN}")
    fi
    echo "[INFO] Lade Modell ${MODEL_URL}"
    echo "[INFO] Ziel ${MODEL_PATH} (ca. 1,9 GB, ein abgebrochener Download wird fortgesetzt)"
    if ! curl -fL --retry 5 --retry-delay 10 --no-progress-meter -C - \
            "${AUTH[@]}" -o "$PART" "$MODEL_URL"; then
        # curl also fails with HTTP 416 if the .part file is already complete
        if [ -z "$MODEL_SHA" ] || ! echo "${MODEL_SHA}  ${PART}" | sha256sum -c --status -; then
            fail "Download fehlgeschlagen."
        fi
    fi
    if [ -n "$MODEL_SHA" ]; then
        echo "[INFO] Prüfe SHA-256"
        if ! echo "${MODEL_SHA}  ${PART}" | sha256sum -c --status -; then
            rm -f "$PART"
            fail "Prüfsumme stimmt nicht, Datei gelöscht. Neustart lädt erneut."
        fi
    fi
    mv "$PART" "$MODEL_PATH"
    echo "[INFO] Download abgeschlossen"
fi

ARGS=(-m "$MODEL_PATH" --host 0.0.0.0 --port 8080 --alias ha-llm
      --ctx-size "$CTX" --parallel 1 --jinja --cache-reuse 256)
if [ "${THREADS:-0}" != "0" ]; then
    ARGS+=(--threads "$THREADS")
fi

echo "[INFO] Starte llama-server: ${ARGS[*]} ${EXTRA}"
# shellcheck disable=SC2086
exec /app/llama-server "${ARGS[@]}" ${EXTRA}
