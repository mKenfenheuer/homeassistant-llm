<img src="brand/logo.svg" alt="HA LLM" height="80">

Local voice assistant components for Home Assistant, running on the GPU of
the Home Assistant host (NVIDIA, AMD, Intel) or on the CPU, without cloud. The
goal is the complete Assist pipeline locally: speech-to-text, conversation and
text-to-speech.

| Component | Part of the pipeline | Location |
|-----------|----------------------|----------|
| App "HA Whisper" (whisper.cpp server) | Speech-to-text | `addons/ha_whisper/` |
| App "HA LLM Inference" (llama.cpp server) | Conversation | `addons/ha_llm_inference/` |
| App "HA TTS" (llama.cpp server with LFM3-Omni or Qwen3-TTS) | Text-to-speech | `addons/ha_tts/` |
| Integration "HA LLM Agent (local)" | Speech-to-text, conversation agent and text-to-speech for Assist | `custom_components/ha_llm_agent/` |
| Model | Conversation | [mkenfenheuer/qwen2.5-3b-homeassistant-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-GGUF) |
| Dataset | Conversation | [mkenfenheuer/homeassistant-tool-calling](https://huggingface.co/datasets/mkenfenheuer/homeassistant-tool-calling) |

The conversation model is Qwen2.5-3B-Instruct with LoRA (Built with Qwen,
Qwen RESEARCH LICENSE, non-commercial use only). It is 1.9 GB (GGUF, Q4_K_M)
and available in German today. The app has a `language` option, so models for
more languages can be added.

## Installation

Every part is optional. Install only the apps you need and pick other
engines for the rest of the Assist pipeline, for example the local Whisper
app for speech-to-text and a cloud conversation agent.

1. **Apps:** In Settings → Apps → App store → ⋮ → Repositories add the URL of
   this repository. Install and start "HA Whisper", "HA LLM Inference"
   and/or "HA TTS".
   On the first start each app downloads its model.
2. **Integration:** In HACS → ⋮ → Custom repositories add the same URL with
   type "Integration". Install "HA LLM Agent (local)" and restart Home
   Assistant.
3. **Setup:** Add "HA LLM Agent (local)" once per app and choose
   "Speech-to-text", "Conversation agent" or "Text-to-speech". The URL of the
   installed app is filled in. In Settings → Voice assistants select
   "Whisper" as speech-to-text, "Local LLM" as conversation agent and "Local
   TTS" as text-to-speech. Conversation agents set up with earlier versions
   are called "Lokales LLM".

The model only sees the entities exposed to Assist under "Expose entities".

## Hardware

Both apps use a GPU of the host if they find one, otherwise the CPU: NVIDIA
through CUDA, AMD and Intel through Vulkan. Vulkan works with today's Home
Assistant OS. NVIDIA needs the NVIDIA variant of Home Assistant OS, so far only
available as a [test image](https://github.com/mKenfenheuer/operating-system/releases).
Both apps share the same base images, Docker stores them only once.

All three apps fit on a GPU with 6 GB: HA LLM Inference needs about 2300 MiB,
HA TTS 2600 MiB and HA Whisper 900 to 1100 MiB. On a smaller GPU the option
`gpu_memory` of HA LLM Inference and HA TTS sets a budget, the rest of the
model then runs on the CPU. If an app doesn't get enough GPU memory at the
start, it tries again and at last runs on the CPU instead of stopping.

Speech-to-text, 11 s of audio:

| Hardware | Whisper model | Time |
|----------|---------------|------|
| GeForce RTX 3090 | large-v3-turbo | 0.15 s |
| CPU (Intel, 8 threads) | small | 3.7 s |
| CPU (Intel, 8 threads) | large-v3-turbo | 15.8 s |

Conversation with 95 entities on 4 cores:

| Hardware | First request | Following commands |
|----------|---------------|--------------------|
| CPU with AVX2 | 34 s | 2–3 s |
| CPU without AVX2 (Atom, Celeron J4125) | 19 min | 30–40 s |
| GTX 1060 6 GB | 4 s | 1–2 s |

An RTX 3090 generates 210 tokens per second, a Radeon RX 6900 XT 130.

Text-to-speech on an RTX 3090, two sentences with 8 s of speech:

| Model | First audio | Complete |
|-------|-------------|----------|
| LFM3-Omni 2B | 0.01–0.04 s | 0.8–1.0 s |
| Qwen3-TTS 1.7B | 1.0 s | 1.9–2.3 s |

Without AVX2 and without a GPU the model should run on another machine. Enter
the URL of that server in the integration under "Reconfigure":

```bash
docker run -d --name ha-llm --restart unless-stopped --gpus all \
  -p 8080:8080 -v $PWD/model:/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-cuda-b12063 \
  -m /models/de/qwen2.5-3b-ha-v18-Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
  --alias ha-llm --jinja -ngl 99 -fa on -c 16384 -np 1 --cache-reuse 256
```

## How it works

**Speech-to-text** The integration collects the recording of the Assist
pipeline and sends it with the language of the pipeline to whisper-server.
Whisper transcribes complete recordings, so the text arrives after the user
stops speaking.

**Text-to-speech** The integration splits the answer into sentences while
the conversation agent is still writing and streams the audio of each
sentence from llama-server (`POST /tts`). Numbers, times and units become
words first ("21,5 °C" → "einundzwanzig Komma fünf Grad Celsius").

**Conversation** The integration sends the exposed entities
(`entity_id | name | area`) and five tools to the model. Together they cover
the built-in Assist intents of Home Assistant:

| Tool | What it does | Example |
|------|--------------|---------|
| `control` | Home Assistant services, for lights also for a whole area | "Licht in der Küche an" → `light.turn_on`, `"area": "Küche"` |
| | music: search and play by artist, track, album, playlist or radio station (Music Assistant or `media_player.search_media`), shuffle, repeat, mute | "Spiel Musik von Queen auf dem Echo" |
| | to-do lists: add, check off, remove | "Setz Milch auf die Einkaufsliste" |
| `get_state` | states and measurements; weather with the course of the rest of the day and the next days; open list items | "Wie ist das Wetter?", "Was steht auf der Einkaufsliste?" |
| `timer` | timers of the voice device: start (also named), cancel, pause, resume, add, subtract, status; delayed commands | "Stell einen Pizza-Timer auf 12 Minuten", "Mach in 10 Minuten das Licht aus" |
| `get_time` | time, weekday and date | "Wie spät ist es?" |
| `broadcast` | announcement on all other voice devices | "Sag allen, dass das Essen fertig ist" |

Timers and broadcasts use the built-in intents of Home Assistant, so they
need a voice device (for example Home Assistant Voice) as the source. The
integration only allows exposed entities and the services in
`llm_schema.SERVICES`.

The option "Answer length" (short, medium, long, detailed) goes into the
system prompt. The model answers accordingly: "Wie ist das Wetter?" gets the
current weather (short), plus the course of the rest of the day (medium),
today's range and rain chance (long), and humidity, wind and tomorrow
(detailed).

If a device reports an error, the model says so instead of claiming success.
This behavior comes from the training data, the integration doesn't enforce it
with rules.

## Results

| Test | Model v18 | Model v17 |
|------|-----------|-----------|
| 300 test conversations in unseen houses: correct tool calls | 98.7 % | 97.0 % |
| same conversations: answers matching the tool result | 100 % | 98.6 % |
| Real house, 60 scored turns incl. seven playlist requests on 76 playlists, covers and windows of the whole house, two questions in one sentence | 57/60 | 53/60 |

Both models were measured on the v18 test conversations. The playlist names
of the real house are not in the training data: v18 picks the fitting one in
6 of 7 requests, v17 in none.

On the same 300 conversations Claude Opus 5.5 (effort low, same prompt and
tools, via the Anthropic API) reaches 90.7 % correct tool calls and 74.0 %
matching answers, with a median of 1.46 s per call. v17 on an RTX 3090 takes
0.40 s.

## Training

```bash
HA_LANG=de ./run_train.sh                              # 27,612 synthetic conversations, continues from the last LoRA adapter, stops when the eval loss no longer drops
HA_LANG=de ./run_export.sh output/de/merged qwen2.5-3b-ha-vX  # GGUF Q4_K_M with importance matrix and chat template
HA_LANG=de ./run_test.sh model/de/qwen2.5-3b-ha-vX-Q4_K_M.gguf  # evaluation on data/de/eval.jsonl
python src/agent_sim.py URL scenarios.json             # scored dialogues against a simulated house
```

`src/gen_dataset.py --lang de` creates every conversation in a newly
randomized house. Each of the 117 cases (intents, question kinds, errors,
conversation turns) gets the same share. `src/soup.py` averages the LoRA
weights of several runs. Training and integration read the same prompt from
`llm_schema.py`. If this file changes, the model must be trained again.

`src/upload_hf.py --lang de --gguf FILE` uploads a model to Hugging Face and
updates `models.json`; `--dataset-only` uploads only the dataset. From then on
all apps with `auto_update` update themselves.

## Languages

Generators exist for German (`de`, released model), English (`en`), Spanish
(`es`), French (`fr`), Italian (`it`) and Dutch (`nl`). Every language has its
own model. Service names, tool names and JSON keys are the same for all
languages; the texts differ:

| Part | Where |
|------|-------|
| System prompt, tool descriptions, answer lengths, weekdays, weather conditions, durations, error texts | `LANGUAGES["<lang>"]` in `custom_components/ha_llm_agent/llm_schema.py` |
| Dataset generator (sentences, grammar, device names) | `src/gen/<lang>/` with a `main(argv)` |
| Consistency check (words that report a failure) | `src/gen/<lang>/check.py` |
| Chat template with the default system prompt | `templates/chat_template.<lang>.jinja` |
| Data, training output, models | `data/<lang>/`, `output/<lang>/`, `model/<lang>/` |
| Release metadata | `hf/model/<lang>/latest.json` |
| App option | `language` in `addons/ha_llm_inference/config.yaml` |

All languages share one model repository on Hugging Face: `models.json` links
each language to its model file (`<lang>/<file>.gguf`). The dataset repository
holds all languages with a `locale` column. The integration offers the
languages in `llm_schema.MODEL_LANGUAGES` (languages with a released model)
and takes the model language from its options, by default the language of
Home Assistant.

```bash
python src/gen_all.py --langs de,en,es,fr,it,nl   # data/<lang>/ for each, merged into data/all/
python src/upload_hf.py --all                      # dataset of all languages to Hugging Face
python src/check_languages.py                      # are all languages in step?
```

**Every change applies to all languages.** A new intent, a fixed answer, a new
device type or a changed tool result goes into every generator in
`src/gen/*/` and, for prompt texts, into every entry of `LANGUAGES`.
`src/check_languages.py` enforces this: all languages need the same text
keys and placeholders, the same case names, and per case the same category
and the same tools and services. The GitHub action `Languages` runs it on
every push together with the consistency check of each generator.

## Limits

- The conversation model is German only so far and only uses the services in
  `llm_schema.SERVICES`. It doesn't create automations, doesn't set alarms
  for a clock time and doesn't answer general knowledge questions.
- Timers and broadcasts need a voice device as the source; typed requests in
  the web interface get "not supported on this device".
- With more than about 250 entities, 8,192 tokens of context aren't enough.
  The integration then shortens the list.
- Speech-to-text isn't streaming yet: whisper.cpp needs the complete
  recording. Text-to-speech streams sentence by sentence.
- Text-to-speech writes numbers and units as words only for German and
  English. Other languages get the text as it is.
