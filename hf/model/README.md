---
license: other
license_name: qwen-research
license_link: LICENSE
language:
  - de
base_model: Qwen/Qwen2.5-3B-Instruct
pipeline_tag: text-generation
tags:
  - gguf
  - home-assistant
  - tool-calling
  - smart-home
  - german
datasets:
  - __DATASET_REPO__
---

# Qwen2.5-3B Home Assistant (German, GGUF)

**Built with Qwen.** LoRA fine-tune of
[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
that controls a Home Assistant installation in German via tool calls and
answers questions about device states. Quantized to Q4_K_M (1.9 GB) for CPU
inference with llama.cpp.

| File | SHA-256 |
|------|---------|
| `__GGUF__` (current) | `__SHA256__` |
| `qwen2.5-3b-ha-v3-Q4_K_M.gguf` (v3, superseded) | `9851d8f0d1495cd058ad5147971fd23851dde20b28e3dd063c40ded694ce649a` |
| `qwen2.5-3b-ha-Q4_K_M.gguf` (v1, superseded) | `134b113ee4914735e6d119bc418ca489f04d55acac0d519510fc6896d81a8467` |

## Usage

The model expects the system prompt and tool schema from
`custom_components/ha_llm_agent/llm_schema.py` of the project repository: a
German instruction, the list of exposed entities (`entity_id | name | area`)
and two tools, `control` (calls Home Assistant services) and `get_state`.
The Home Assistant integration and add-on in the project repository build this
prompt automatically.

```bash
llama-server -m __GGUF__ --jinja -c 8192 --port 8080
```

## Training

- Data: [__DATASET_REPO__](https://huggingface.co/datasets/__DATASET_REPO__),
  8,000 synthetic German conversations over randomly generated houses
  (up to 110 entities, group entities, rooms without lights, lights named
  after their room, several irrigation devices, brand names, devices that are
  not in the list, unreachable devices, multi-turn conversations).
- v4 (current): LoRA r=32, alpha=64, all linear layers, 2.5 epochs from
  Qwen2.5-3B-Instruct, lr 2e-4, loss on assistant tokens only. RTX 3090,
  4.8 hours.
- v1 to v3 used the same setup on earlier data; v2 was trained on top of v1.

## Honest answers

The `control` tool returns `{"success": false, "error": "...", "done": [...]}`
when a device is unavailable. The training data teaches the model to report
this instead of claiming success, including in later turns ("Ist es jetzt
an?", "Versuch es nochmal"). Every training answer is checked against the
preceding tool results; no answer claims a success the tool did not report.

## Evaluation

300 held-out conversations (v4 data) in houses not seen during training:

| Model | Correct tool calls | Grounded answers | Commands | Errors reported |
|-------|-------------------|------------------|----------|-----------------|
| v3 | 93.3 % | 96.8 % | 111/119 | 8/10 |
| v4 (current) | 96.0 % | 100 % | 117/119 | 10/10 |

Most remaining errors are long lists ("Welche Lampen sind eingeschaltet?" or "Alle
Lichter aus" in houses with many lights), where the model miscopies or skips one entity_id.

On a real house with over 100 exposed entities, v4 handles all 31 acceptance
sentences correctly (v3: 30). This includes "Mach das Licht in der Küche an"
with lights named "Küche", "Küche Links" and "Küche Rechts", and "Bewässere
die Balkonpflanzen" next to a garden irrigation button.

In 18 simulated multi-turn dialogues with unreachable devices, v4 never
claims success for a device that failed. After a partial success it names
both parts, and "Was hast du eingeschaltet?" lists only what was switched (v3
also listed the unreachable lamp). Known error: after "Licht in der Küche an"
the correction "Nein, ich meinte im Esszimmer" turns off only one of the four
kitchen lights.

End-to-end in Home Assistant 2026.10.4 (demo devices with German names,
including an unavailable lamp and a follow-up turn): 24/24 test sentences,
median latency 3.0 s on 8 CPU threads.

## Limitations

- German only; supports the services listed in `llm_schema.SERVICES`.
- No timers, automations, weather forecasts or general knowledge questions.
- Above ~250 exposed entities the 8k context is not sufficient.

## License

Derivative of Qwen2.5-3B-Instruct, distributed under the
[Qwen RESEARCH LICENSE AGREEMENT](LICENSE): **non-commercial use only**.
Modifications: LoRA fine-tuning (merged), conversion to GGUF and Q4_K_M
quantization. See `NOTICE`.
