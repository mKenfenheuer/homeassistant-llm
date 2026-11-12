# Changelog

## 0.1.1

- If another app holds the GPU memory, the app waits, tries again three times
  and then runs on the CPU, instead of stopping.

## 0.1.0

- First version: whisper.cpp 1.9.5 on the llama.cpp b12063 images with CUDA,
  Vulkan and CPU, Whisper models from Hugging Face with SHA-256 check.
- Warm-up with one second of silence after the start, and a CUDA cache in
  `/data` that survives restarts. On a GTX 1060 the first request after a
  restart takes 1.4 s instead of 15.7 s.
