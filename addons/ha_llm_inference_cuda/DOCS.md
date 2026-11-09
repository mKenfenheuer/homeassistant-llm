# HA LLM Inference (NVIDIA)

Dieses Add-on startet `llama-server` (llama.cpp) mit dem feinabgestimmten
Modell [qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF)
auf einer NVIDIA-GPU über CUDA 12.8. Modell, automatische Updates und Optionen entsprechen dem Add-on
„HA LLM Inference“ (CPU). Es lädt das Modell ebenfalls nach `/share/ha_llm/`.

## Voraussetzungen

Das Add-on braucht Home Assistant OS mit GPU-Unterstützung für Apps
(Option `gpu` im Supervisor). Diese ist noch nicht offiziell veröffentlicht,
Testimages gibt es unter
[mKenfenheuer/operating-system](https://github.com/mKenfenheuer/operating-system/releases).
Ohne GPU-Zugriff läuft das Modell auf der CPU, das Log zeigt dann eine
Warnung.

Unterstützte Hardware: NVIDIA-GPUs ab Maxwell (GeForce GTX 900) mit der Treibererweiterung `nvidia` (Turing und neuer) oder `nvidia-legacy` (Maxwell bis Volta). Getestet mit einer GeForce GTX 1060.

## Integration

Die Integration „HA LLM Agent (lokal)“ findet das Add-on selbst. Der Hostname
steht auf der Info-Seite des Add-ons, zum Beispiel
`http://<präfix>-ha-llm-inference-cuda:8080`. Sind CPU- und GPU-Add-on
installiert, nimmt die Integration bei der Einrichtung das GPU-Add-on. Eine
bestehende Integration lässt sich über „Neu konfigurieren“ umstellen.

## GPU-Schichten

`gpu_layers` (Standard 99) legt fest, wie viele Schichten des Modells auf der
GPU rechnen. Das Modell hat 36 Schichten und braucht mit 8192 Tokens Kontext
etwa 2,5 GB Grafikspeicher. Mit 0 rechnet nur die CPU.

## Lizenz

Das Modell basiert auf Qwen2.5-3B-Instruct (Qwen RESEARCH LICENSE AGREEMENT)
und ist nur für nicht-kommerzielle Nutzung freigegeben. Built with Qwen.
