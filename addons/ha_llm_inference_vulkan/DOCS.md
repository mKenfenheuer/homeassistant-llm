# HA LLM Inference (AMD/Intel)

Dieses Add-on startet `llama-server` (llama.cpp) mit dem feinabgestimmten
Modell [qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF)
auf einer AMD- oder Intel-GPU über Vulkan (Mesa: RADV, ANV). Modell, automatische Updates und Optionen entsprechen dem Add-on
„HA LLM Inference“ (CPU). Es lädt das Modell ebenfalls nach `/share/ha_llm/`.

## Voraussetzungen

Das Add-on läuft auf jedem amd64- und aarch64-System. Findet llama.cpp keine
GPU, rechnet das Modell auf der CPU, das Log zeigt das beim Start.

Das Add-on bekommt die GPU über die Option `video` (`/dev/dri`). Das reicht für
Vulkan auf AMD und Intel, auch mit dem heutigen Home Assistant OS.

Unterstützte Hardware: AMD-GPUs mit dem Treiber `amdgpu` (GCN 3 und neuer) und Intel-GPUs mit `i915` oder `xe`. Die Treiber stecken im Kernel, eine Erweiterung ist nicht nötig. Getestet mit einer Radeon RX 6900 XT. Auf aarch64 nutzt llama.cpp Vulkan, wenn Mesa die GPU unterstützt (etwa V3D auf dem Raspberry Pi 5), ungetestet.

## Integration

Die Integration „HA LLM Agent (lokal)“ findet das Add-on selbst. Der Hostname
steht auf der Info-Seite des Add-ons, zum Beispiel
`http://<präfix>-ha-llm-inference-vulkan:8080`. Sind CPU- und GPU-Add-on
installiert, nimmt die Integration bei der Einrichtung das GPU-Add-on. Eine
bestehende Integration lässt sich über „Neu konfigurieren“ umstellen.

## GPU-Schichten

`gpu_layers` (Standard 99) legt fest, wie viele Schichten des Modells auf der
GPU rechnen. Das Modell hat 36 Schichten und braucht mit 8192 Tokens Kontext
etwa 2,5 GB Grafikspeicher. Mit 0 rechnet nur die CPU.

## Lizenz

Das Modell basiert auf Qwen2.5-3B-Instruct (Qwen RESEARCH LICENSE AGREEMENT)
und ist nur für nicht-kommerzielle Nutzung freigegeben. Built with Qwen.
