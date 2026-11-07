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
| `qwen2.5-3b-ha-v7-Q4_K_M.gguf` (v7, superseded) | `507d2f74f64cf63df4a0bca5098c2cae480cecab1358902b05e61703b9e94592` |
| `qwen2.5-3b-ha-v4-Q4_K_M.gguf` (v4, superseded) | `1235a58736df7397e96271ed4cec2313fa5bd7ab436bfc1d58c2097b5d1af57a` |
| `qwen2.5-3b-ha-v3-Q4_K_M.gguf` (v3, superseded) | `9851d8f0d1495cd058ad5147971fd23851dde20b28e3dd063c40ded694ce649a` |
| `qwen2.5-3b-ha-Q4_K_M.gguf` (v1, superseded) | `134b113ee4914735e6d119bc418ca489f04d55acac0d519510fc6896d81a8467` |

## Usage

The model expects the system prompt and tool schema from
`custom_components/ha_llm_agent/llm_schema.py` of the project repository: a
German instruction, the list of exposed entities (`entity_id | name | area`)
and two tools, `control` (calls Home Assistant services) and `get_state`.
Since v11 (schema 5) both tools accept an `area` for lights ("Mach das Licht
in der Küche an" → `{"service": "light.turn_on", "area": "Küche"}`, `"*"` for
the whole house); the integration resolves the area to entity_ids. v11 needs
integration 0.7.0 or later. The Home Assistant integration and add-on in the project repository build this
prompt automatically.

```bash
llama-server -m __GGUF__ --jinja -c 8192 --port 8080
```

## Training

- Data: [__DATASET_REPO__](https://huggingface.co/datasets/__DATASET_REPO__),
  8,000 synthetic German conversations over randomly generated houses
  (up to 110 entities, group entities, rooms without lights, lights named
  after their room, several irrigation devices, brand names, devices that are
  not in the list, unreachable devices, rooms where only some lights are
  reachable, users insisting on missing devices, heating mode switches,
  door contacts, electric cars, outdoor temperature next to forecast and
  weather sensors, area targeting, multi-turn conversations). Each of the 64
  generator cases has the same quota, rare cases included.
- v11 (current): LoRA r=32, alpha=64, all linear layers, 2.5 epochs from
  Qwen2.5-3B-Instruct, lr 2e-4, loss on assistant tokens only. RTX 3090,
  5.5 hours.
- v1 to v7 used the same setup on earlier data; v2 was trained on top of v1.
  v5, v6 and v8 to v10 were trained but not released (regressions on the
  real house).

## Honest answers

The `control` tool returns `{"success": false, "error": "...", "done": [...]}`
when a device is unavailable. The training data teaches the model to report
this instead of claiming success, including in later turns ("Ist es jetzt
an?", "Versuch es nochmal"). Every training answer is checked against the
preceding tool results; no answer claims a success the tool did not report.

## Evaluation

300 held-out conversations in houses not seen during training. v11 data use
schema 5 (area targeting) and add outdoor, forecast and door questions, so the
two rows are not directly comparable:

| Model | Test data | Correct tool calls | Grounded answers | Missing devices refused | State questions |
|-------|-----------|-------------------|------------------|-------------------------|-----------------|
| v7 | v7 | 98.0 % | 100 % | 31/31 | 40/40 |
| v11 (current) | v11 | 98.7 % | 98.7 % | 15/15 | 78/81 |

On a real house with over 100 exposed entities, the acceptance test covers 26
sentences on an older and 32 sentences on the current snapshot of the device
list, plus four variants of the current list with 8 devices removed each:

| Model | Older snapshot | Current snapshot | Variants |
|-------|----------------|------------------|----------|
| v7 | 24/26 | 32/32 | 123/128 |
| v11 (current) | 26/26 | 32/32 | 128/128 |

v7 queried the front door for "Ist die Balkontür im Wohnzimmer offen?" and the
charge level when asked for the range of the electric car; v11 answers both
correctly.

In 19 simulated multi-turn dialogues with unreachable devices, v11 never
claims success for a device that failed. With two of four kitchen lights
unreachable, "Mach das Licht in der Küche an" switches the room by area and
names the switched and the unreachable lights. The correction "Nein, ich
meinte im Esszimmer" after "Licht in der Küche an" now turns off all kitchen
lights (v7: only one). Known weaknesses: asked for a TV in a bedroom that
only holds a voice satellite, v11 correctly switches nothing but names the
satellite's media player instead of the room. In the first answer about the
partly unreachable kitchen it names only one of the two unreachable lights.

End-to-end in Home Assistant 2026.10.4 (demo devices with German names,
including an unavailable lamp and a follow-up turn): 24/24 test sentences
with v11, median latency 3.2 s on 8 CPU threads.

## Limitations

- German only; supports the services listed in `llm_schema.SERVICES`.
- No timers, automations or general knowledge questions. Weather forecasts
  only through exposed forecast sensors.
- Above ~250 exposed entities the 8k context is not sufficient.

## License

Derivative of Qwen2.5-3B-Instruct, distributed under the
[Qwen RESEARCH LICENSE AGREEMENT](LICENSE): **non-commercial use only**.
Modifications: LoRA fine-tuning (merged), conversion to GGUF and Q4_K_M
quantization. See `NOTICE`.
