# HA LLM Inference

Dieses Add-on startet `llama-server` (llama.cpp) mit dem feinabgestimmten
Modell [qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF).
Die Integration „HA LLM Agent (lokal)“ spricht den Server unter
`http://local-ha-llm-inference:8080` an. Bei einem Add-on aus einem
Git-Repository lautet der Hostname `<präfix>-ha-llm-inference`. Er steht auf der
Info-Seite des Add-ons.

## Modell

Beim ersten Start lädt das Add-on die GGUF-Datei (1,9 GB) von Hugging Face nach
`/share/ha_llm/` und prüft die SHA-256-Summe. Ein abgebrochener Download wird
beim nächsten Start fortgesetzt. Danach startet der Server ohne Download.

Mit `auto_update` (Standard: an) prüft das Add-on beim Start und alle
`update_interval` Stunden (Standard: 6), ob auf Hugging Face ein neues Modell
liegt (`latest.json`). Ein neues Modell lädt es im laufenden Betrieb, prüft
die SHA-256-Summe, startet den Server damit neu und löscht das alte. Die
erste Anfrage danach dauert so lange wie nach einem Neustart. Für ein neues
Modell ist zeitweise doppelter Speicherplatz nötig (2 × 1,9 GB).

Ein eigenes Modell lässt sich über `model_path` oder `model_url` einbinden, die automatischen Updates sind dann aus. Beispiel:
`/share/ha_llm/eigenes.gguf`. Für ein privates Hugging-Face-Repository das
Token unter `hf_token` eintragen.

## GPU

Das Add-on nutzt eine GPU des Hosts, wenn es eine findet, sonst die CPU. Das
Log zeigt beim Start, welche:

- **NVIDIA** (amd64, ab Maxwell) über CUDA. Der Host braucht den NVIDIA-Treiber:
  Home Assistant OS in der Variante `nvidia` (Turing und neuer) oder
  `nvidia-legacy` (Maxwell bis Volta). Diese ist noch nicht offiziell
  veröffentlicht, Testimages gibt es unter
  [mKenfenheuer/operating-system](https://github.com/mKenfenheuer/operating-system/releases).
- **AMD und Intel** über Vulkan (Mesa). Die Treiber stecken im Kernel, das
  funktioniert auch mit dem heutigen Home Assistant OS.
- **ARM** (aarch64) über Vulkan, wenn Mesa die GPU unterstützt (etwa V3D auf
  dem Raspberry Pi 5), ungetestet.

Das Add-on bekommt die GPU über die Option `video`. Mit einer NVIDIA-GPU nimmt
es CUDA, sonst Vulkan.

`gpu_layers` (Standard 99) legt fest, wie viele Schichten des Modells auf der
GPU rechnen. Das Modell hat 36 Schichten und braucht mit 8192 Tokens Kontext
etwa 2,5 GB Grafikspeicher. Mit 0 rechnet nur die CPU.

Getestet mit einer GeForce RTX 3090 (CUDA, 210 Tokens pro Sekunde) und einer
Radeon RX 6900 XT (Vulkan, 130 Tokens pro Sekunde).

## Lizenz

Das Modell basiert auf Qwen2.5-3B-Instruct (Qwen RESEARCH LICENSE AGREEMENT)
und ist nur für nicht-kommerzielle Nutzung freigegeben. Built with Qwen.
