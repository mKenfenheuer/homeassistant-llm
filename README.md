<img src="brand/logo.svg" alt="HA LLM" height="80">

Ein lokales deutsches Sprachmodell für Home Assistant. Es schaltet Geräte und
beantwortet Fragen zu ihrem Zustand, ohne Cloud. Basis ist
Qwen2.5-3B-Instruct mit LoRA (Built with Qwen, Qwen RESEARCH LICENSE, nur
nicht-kommerzielle Nutzung). Das Modell ist 1,9 GB groß (GGUF, Q4_K_M) und
läuft auf der CPU des Home-Assistant-Hosts oder auf einer GPU.

| Komponente | Ort |
|------------|-----|
| Add-on „HA LLM Inference“ (llama.cpp-Server, CPU) | `addons/ha_llm_inference/` |
| Add-on „HA LLM Inference (NVIDIA)“ (CUDA, experimentell) | `addons/ha_llm_inference_cuda/` |
| Add-on „HA LLM Inference (AMD/Intel)“ (Vulkan, experimentell) | `addons/ha_llm_inference_vulkan/` |
| Integration „HA LLM Agent (lokal)“ (Gesprächsagent für Assist) | `custom_components/ha_llm_agent/` |
| Modell | [mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF) |
| Datensatz | [mkenfenheuer/homeassistant-tool-calling-de](https://huggingface.co/datasets/mkenfenheuer/homeassistant-tool-calling-de) |

## Installation

1. **Add-on:** Unter Einstellungen → Add-ons → Add-on-Store → ⋮ →
   Repositories die URL dieses Repositorys eintragen. Danach „HA LLM
   Inference“ installieren und starten. Beim ersten Start lädt das Add-on
   das Modell. Danach prüft es alle 6 Stunden, ob ein neues Modell vorliegt
   (`auto_update`).
2. **Integration:** In HACS → ⋮ → Benutzerdefinierte Repositories dieselbe
   URL mit Typ „Integration“ eintragen. Danach „HA LLM Agent (lokal)“
   installieren und Home Assistant neu starten.
3. **Einrichtung:** „HA LLM Agent (lokal)“ hinzufügen. Die URL des Add-ons
   ist vorausgefüllt. Unter Einstellungen → Sprachassistenten „Lokales LLM“
   als Gesprächsagenten wählen.

Das Modell sieht nur die Entitäten, die unter „Entitäten freigeben“ für
Assist freigegeben sind.

## Hardware

Das Add-on braucht eine CPU mit AVX2, z. B. einen Intel N100 oder Ryzen.
Gemessen mit 95 Entitäten auf 4 Kernen:

| Rechner | Erste Anfrage | Folgende Befehle |
|---------|---------------|------------------|
| CPU mit AVX2 | 34 s | 2–3 s |
| CPU ohne AVX2 (Atom, Celeron J4125) | 19 min | 30–40 s |
| GTX 1060 6 GB | 4 s | 1–2 s |

Mit einer GPU im Home-Assistant-Host rechnet das Modell dort: „HA LLM
Inference (NVIDIA)“ für NVIDIA-GPUs, „HA LLM Inference (AMD/Intel)“ für AMD-
und Intel-GPUs. Beide brauchen Home Assistant OS mit GPU-Unterstützung für
Apps, die es bisher nur als
[Testimage](https://github.com/mKenfenheuer/operating-system/releases) gibt.
Auf einer Radeon RX 6900 XT erzeugt das Vulkan-Add-on 130 Tokens pro Sekunde.

Ohne AVX2 und ohne GPU sollte das Modell auf einem anderen Rechner laufen. Die URL des
Servers trägt man dann in der Integration unter „Neu konfigurieren“ ein:

```bash
docker run -d --name ha-llm --restart unless-stopped --gpus all \
  -p 8080:8080 -v $PWD/model:/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12063 \
  -m /models/qwen2.5-3b-ha-v15.1-Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
  --alias ha-llm --jinja -ngl 99 -fa on -c 16384 -np 1 --cache-reuse 256
```

## Funktionsweise

Die Integration schickt dem Modell die freigegebenen Entitäten
(`entity_id | Name | Bereich`) und zwei Werkzeuge. `control` ruft
Home-Assistant-Dienste auf, bei Lichtern auch für einen ganzen Bereich
(„Licht in der Küche an“ → `"area": "Küche"`). `get_state` liest Zustände und
Messwerte. Die Integration lässt nur freigegebene Entitäten und die Dienste
aus `llm_schema.SERVICES` zu.

Meldet ein Gerät einen Fehler, sagt das Modell das, statt einen Erfolg zu
behaupten. Das Verhalten stammt aus den Trainingsdaten, die Integration
erzwingt es nicht durch Regeln.

## Ergebnisse

| Test | Modell v15.1 |
|------|--------------|
| 300 Testgespräche in unbekannten Häusern: korrekte Werkzeugaufrufe | 99,7 % |
| dieselben Gespräche: Antworten passend zum Werkzeugergebnis | 100 % |
| Echtes Haus mit über 100 Entitäten: 58 Sätze plus 4 Varianten | 186/186 |
| Home Assistant Ende zu Ende (CPU, 8 Threads) | 24/24, Median 2,9 s |

## Training

```bash
./run_train.sh                                  # 8.000 synthetische Gespräche, LoRA r=32, ~5,5 h auf RTX 3090
./run_export.sh output/merged qwen2.5-3b-ha-vX  # GGUF Q4_K_M mit Importance Matrix und Chat-Vorlage
./run_test.sh                                   # Bewertung auf data/eval.jsonl
```

`src/gen_dataset.py` erzeugt jedes Gespräch in einem neu gewürfelten Haus.
`src/soup.py` mittelt die LoRA-Gewichte mehrerer Läufe. v15.1 ist das Mittel
aus drei Läufen. Training und Integration lesen denselben Prompt aus
`llm_schema.py`. Ändert sich diese Datei, muss das Modell neu trainiert
werden.

`src/upload_hf.py` lädt ein neues Modell auf Hugging Face hoch und schreibt
`latest.json`. Ab diesem Moment aktualisieren sich alle Add-ons mit
`auto_update`.

## Grenzen

- Nur Deutsch, nur die Dienste in `llm_schema.SERVICES`. Das Modell legt
  keine Timer oder Automationen an und beantwortet keine Wissensfragen.
- Bei mehr als etwa 250 Entitäten reichen 8.192 Tokens Kontext nicht. Die
  Integration kürzt dann die Liste.
