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
| Model | Conversation | [mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF) |
| Dataset | Conversation | [mkenfenheuer/homeassistant-tool-calling-de](https://huggingface.co/datasets/mkenfenheuer/homeassistant-tool-calling-de) |

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
  -m /models/qwen2.5-3b-ha-v15.1-Q4_K_M.gguf --host 0.0.0.0 --port 8080 \
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
(`entity_id | name | area`) and two tools to the model. `control` calls Home
Assistant services, for lights also for a whole area ("Licht in der Küche an"
→ `"area": "Küche"`). `get_state` reads states and measurements. The
integration only allows exposed entities and the services in
`llm_schema.SERVICES`.

If a device reports an error, the model says so instead of claiming success.
This behavior comes from the training data, the integration doesn't enforce it
with rules.

## Results

| Test | Model v15.1 |
|------|-------------|
| 300 test conversations in unseen houses: correct tool calls | 99.7 % |
| same conversations: answers matching the tool result | 100 % |
| Real house with more than 100 entities: 58 sentences plus 4 variants | 186/186 |
| Home Assistant end to end (CPU, 8 threads) | 24/24, median 2.9 s |

## Training

```bash
./run_train.sh                                  # 8,000 synthetic conversations, LoRA r=32, ~5.5 h on an RTX 3090
./run_export.sh output/merged qwen2.5-3b-ha-vX  # GGUF Q4_K_M with importance matrix and chat template
./run_test.sh                                   # evaluation on data/eval.jsonl
```

`src/gen_dataset.py` creates every conversation in a newly randomized house.
`src/soup.py` averages the LoRA weights of several runs. v15.1 is the average
of three runs. Training and integration read the same prompt from
`llm_schema.py`. If this file changes, the model must be trained again.

`src/upload_hf.py` uploads a new model to Hugging Face and writes
`latest.json`. From then on all apps with `auto_update` update themselves.

## Limits

- The conversation model is German only and only uses the services in
  `llm_schema.SERVICES`. It doesn't create timers or automations and doesn't
  answer general knowledge questions. A model for another language also needs
  the prompt of the integration in that language.
- With more than about 250 entities, 8,192 tokens of context aren't enough.
  The integration then shortens the list.
- Speech-to-text isn't streaming yet: whisper.cpp needs the complete
  recording. Text-to-speech streams sentence by sentence.
- Text-to-speech writes numbers and units as words only for German and
  English. Other languages get the text as it is.
