# HA LLM Inference

This app runs `llama-server` (llama.cpp) with the fine-tuned Home Assistant
model [qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF).
The "HA LLM Agent (local)" integration talks to the server at
`http://local-ha-llm-inference:8080`. For an app from a Git repository the
host name is `<prefix>-ha-llm-inference`, shown on the info page of the app.

## Language

`language` selects the language of the model. Every language has its own
model on Hugging Face (`mkenfenheuer/qwen2.5-3b-homeassistant-<language>-GGUF`),
the app downloads the one of the selected language. Available today: German
(`de`). German models stay in `/share/ha_llm/`, other languages go to
`/share/ha_llm/<language>/`.

## Model

On the first start the app downloads the GGUF file (1.9 GB) from Hugging Face
to `/share/ha_llm/` and checks its SHA-256. An interrupted download resumes on
the next start. After that the server starts without a download.

With `auto_update` (default: on) the app checks on start and every
`update_interval` hours (default: 6) whether Hugging Face has a new model
(`latest.json`). It downloads a new model while the server keeps running,
checks the SHA-256, restarts the server with it and deletes the old one. The
first request afterwards takes as long as after a restart. A new model
temporarily needs twice the disk space (2 × 1.9 GB).

A custom model can be set with `model_path` or `model_url`, automatic updates
are off then. Example: `/share/ha_llm/custom.gguf`. For a private Hugging Face
repository enter the token in `hf_token`.

## GPU

The app uses a GPU of the host if it finds one, otherwise the CPU. The log
shows which one on start:

- **NVIDIA** (amd64, Maxwell and newer) through CUDA. The host needs the
  NVIDIA driver: Home Assistant OS in the `nvidia` (Turing and newer) or
  `nvidia-legacy` (Maxwell to Volta) variant. It isn't officially released
  yet, test images are at
  [mKenfenheuer/operating-system](https://github.com/mKenfenheuer/operating-system/releases).
- **AMD and Intel** through Vulkan (Mesa). The drivers are part of the
  kernel, this works with today's Home Assistant OS.
- **ARM** (aarch64) through Vulkan, if Mesa supports the GPU (like V3D on the
  Raspberry Pi 5), untested.

The app gets the GPU through the `video` option. With an NVIDIA GPU it uses
CUDA, otherwise Vulkan.

`gpu_layers` (default 99) sets how many layers of the model run on the GPU.
The model has 36 layers and needs about 2.5 GB of GPU memory with a context of
8192 tokens. With 0 only the CPU computes.

Tested with a GeForce RTX 3090 (CUDA, 210 tokens per second) and a Radeon
RX 6900 XT (Vulkan, 130 tokens per second).

## License

The model is based on Qwen2.5-3B-Instruct (Qwen RESEARCH LICENSE AGREEMENT)
and only licensed for non-commercial use. Built with Qwen.
