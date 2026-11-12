# HA Whisper

This app runs `whisper-server` (whisper.cpp) for speech-to-text in Assist.
The "HA LLM Agent (local)" integration talks to the server at
`http://local-ha-whisper:8080`. For an app from a Git repository the host
name is `<prefix>-ha-whisper`, shown on the info page of the app.

## Setup

1. Install and start the app. On the first start it downloads the Whisper
   model to `/share/ha_whisper/`.
2. Add the "HA LLM Agent (local)" integration and choose "Speech-to-text".
   The URL of the app is filled in.
3. In Settings → Voice assistants select "Whisper" as speech-to-text engine
   of the pipeline.

## Model and language

`model` selects the Whisper model from the
[whisper.cpp repository](https://huggingface.co/ggerganov/whisper.cpp) on
Hugging Face. The app checks the SHA-256 that Hugging Face reports and
removes models it doesn't use anymore.

| Model | Size | Use |
|-------|------|-----|
| `auto` (default) | | `large-v3-turbo` on a GPU, `small` on the CPU |
| `tiny`, `base` | 32 MB, 60 MB | slow CPUs, lower accuracy |
| `small` | 190 MB | CPU |
| `medium` | 540 MB | fast CPUs or GPU |
| `large-v3-turbo` | 570 MB | GPU, best balance of speed and accuracy |
| `large-v3` | 1.1 GB | GPU, highest accuracy, slower |

All models are multilingual. The integration sends the language of the Assist
pipeline with every request, so one model serves all languages. `language`
(default `auto`) is only used if a request has no language.

## GPU

The app uses a GPU of the host if it finds one, otherwise the CPU. The log
shows which one on start:

- **NVIDIA** (amd64, Maxwell and newer) through CUDA. The host needs the
  NVIDIA driver: Home Assistant OS in the `nvidia` (Turing and newer) or
  `nvidia-legacy` (Maxwell to Volta) variant, so far only available as
  [test image](https://github.com/mKenfenheuer/operating-system/releases).
- **AMD and Intel** through Vulkan (Mesa), with today's Home Assistant OS.
- **ARM** (aarch64) through Vulkan, if Mesa supports the GPU, untested.

The app gets the GPU through the `video` option. Switch `gpu` off to keep the
GPU free, for example for the language model; speech recognition then runs on
the CPU.

With `large-v3-turbo` the app needs about 900 to 1100 MiB of GPU memory. If
another app holds the GPU memory, loading fails. The app then waits until the
other app has finished loading and tries again, three times, and then runs on
the CPU. A restart of the app tries the GPU again.

Tested with a GeForce RTX 3090: 11 s of audio take 0.15 s with
`large-v3-turbo`. On the CPU (8 threads) the same audio takes 3.7 s with
`small`.

## Limits

whisper.cpp transcribes complete recordings. The text arrives after the user
stops speaking, there are no partial results while speaking.

## License

whisper.cpp is MIT licensed, the Whisper models by OpenAI are MIT licensed.
