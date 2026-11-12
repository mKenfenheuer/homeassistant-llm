# Changelog

## 0.2.0

- Fixed context of 4096 tokens. Without it llama-server took the 128k tokens
  of the model and all free GPU memory, if the app started before the others.
- New option `gpu_memory`: a budget for the GPU memory in MiB.
- If another app holds the GPU memory, the app waits, tries again three times,
  then starts with fewer layers on the GPU and at last on the CPU.

## 0.1.0

- First version: llama-server of llama.cpp b12063 with `POST /tts` on CUDA,
  Vulkan or the CPU. Models LFM3-Omni 2B (19 languages, built-in voices) and
  Qwen3-TTS 1.7B (10 languages, voice of a reference recording in
  `/share/ha_tts/voices/`), downloaded from Hugging Face with SHA-256 check.
- Integration 0.9.0: new entry type "Text-to-speech". It streams the answer
  sentence by sentence and writes numbers, times and units as words.
