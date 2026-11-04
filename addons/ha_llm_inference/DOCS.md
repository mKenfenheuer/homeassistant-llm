# HA LLM Inference

Dieses Add-on startet `llama-server` (llama.cpp, CPU) mit dem feinabgestimmten
Modell [qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF).
Die Integration „HA LLM Agent (lokal)“ spricht den Server unter
`http://local-ha-llm-inference:8080` an. Bei einem Add-on aus einem
Git-Repository lautet der Hostname `<präfix>-ha-llm-inference`. Er steht auf der
Info-Seite des Add-ons.

## Modell

Beim ersten Start lädt das Add-on die GGUF-Datei (1,9 GB) von Hugging Face nach
`/share/ha_llm/` und prüft die SHA-256-Summe. Ein abgebrochener Download wird
beim nächsten Start fortgesetzt. Danach startet der Server ohne Download.

Ein eigenes Modell lässt sich über `model_path` einbinden, z. B.
`/share/ha_llm/eigenes.gguf`. Für ein privates Hugging-Face-Repository das
Token unter `hf_token` eintragen.

## Lizenz

Das Modell basiert auf Qwen2.5-3B-Instruct (Qwen RESEARCH LICENSE AGREEMENT)
und ist nur für nicht-kommerzielle Nutzung freigegeben. Built with Qwen.
