# Changelog

## 0.9.1

- Läuft auf jedem amd64- und aarch64-System: ohne GPU rechnet das Modell auf
  der CPU. Das Add-on ist nicht mehr experimentell.
- GPU-Zugriff über die Option `video` statt `gpu`. Vulkan auf AMD und Intel
  funktioniert damit auch mit dem heutigen Home Assistant OS.

## 0.9.0

- Erste Version: llama.cpp b12063 mit Vulkan, Modell und
  automatische Updates wie im CPU-Add-on 0.8.1.
