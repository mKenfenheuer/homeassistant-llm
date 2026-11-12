# HA TTS

This app runs `llama-server` (llama.cpp) with a text-to-speech model for
Assist. The "HA LLM Agent (local)" integration talks to the server at
`http://local-ha-tts:8080`. For an app from a Git repository the host name is
`<prefix>-ha-tts`, shown on the info page of the app.

## Setup

1. Install and start the app. On the first start it downloads the model to
   `/share/ha_tts/`.
2. Add the "HA LLM Agent (local)" integration and choose "Text-to-speech".
   The URL of the app is filled in.
3. In Settings → Voice assistants select "Local TTS" as text-to-speech engine
   of the pipeline and pick a voice.

## Models and voices

`model` selects the model. Both come from the
[ggml-org](https://huggingface.co/ggml-org) repositories on Hugging Face, the
app checks the SHA-256 that Hugging Face reports.

| Model | Size | Languages | Voices |
|-------|------|-----------|--------|
| `lfm3-omni` (default) | 1.6 GB | ar, da, de, el, en, es, fi, fr, he, it, ja, ko, nl, pl, pt, sv, tr, zh | built-in `female` and `male` |
| `qwen3-tts` | 1.5 GB | de, en, es, fr, it, ja, ko, pt, ru, zh | `default`, plus every recording in `/share/ha_tts/voices/` |

With `qwen3-tts` the model speaks with the voice of a reference recording.
Put a WAV or MP3 file of 5 to 15 seconds of clear speech into
`/share/ha_tts/voices/`, for example `max.wav`. It shows up as voice "max" in
the pipeline. Without a recording (voice `default`) the voice changes from
sentence to sentence.

The pipeline sends its language with every request, so one model serves all
of its languages.

## Speed and streaming

The integration splits the answer into sentences while the conversation agent
is still writing it and streams the audio of every sentence. Measured on a
GeForce RTX 3090, two sentences with 8 s of speech:

| Model | First audio | Complete |
|-------|-------------|----------|
| `lfm3-omni` | 0.01–0.04 s | 0.8–1.0 s |
| `qwen3-tts` | 1.0 s | 1.9–2.3 s |

Numbers, times and units are written out before speaking ("21,5 °C" →
"einundzwanzig Komma fünf Grad Celsius") for German and English, because the
models misread digits and symbols.

## GPU

The app uses a GPU of the host if it finds one, otherwise the CPU: NVIDIA
through CUDA (needs the `nvidia` or `nvidia-legacy` variant of Home Assistant
OS), AMD and Intel through Vulkan. It gets the GPU through the `video` option.
Switch `gpu` off to keep the GPU free; speech synthesis then runs on the CPU.

LFM3-Omni needs about 2600 MiB of GPU memory. `gpu_memory` (MiB, default 0 =
no limit) is a budget: the app keeps at most this much and puts the remaining
layers on the CPU. The budgets of all apps together must fit into the GPU
memory.

If another app holds the GPU memory, loading fails. The app then waits until
the other app has finished loading and tries again, three times. After that it
starts with fewer layers on the GPU and at last on the CPU, and says so in the
log. A restart of the app tries the GPU again.

## Licenses

LFM3-Omni: LFM Open License v1.0 by Liquid AI. Qwen3-TTS: Apache 2.0. Check
the model cards on Hugging Face before commercial use.
