# Changelog

## 0.1.0

- First version: llama-server of llama.cpp b12063 with `POST /tts` on CUDA,
  Vulkan or the CPU. Models LFM3-Omni 2B (19 languages, built-in voices) and
  Qwen3-TTS 1.7B (10 languages, voice of a reference recording in
  `/share/ha_tts/voices/`), downloaded from Hugging Face with SHA-256 check.
- Integration 0.9.0: new entry type "Text-to-speech". It streams the answer
  sentence by sentence and writes numbers, times and units as words.
