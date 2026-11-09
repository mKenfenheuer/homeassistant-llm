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
answers questions about device states. Quantized to Q4_K_M (1.9 GB) with an
importance matrix. Runs on a CPU with AVX2 (about 16 tokens/s output on
4 cores) or on small GPUs (GTX 1060 6 GB: 540 tokens/s prompt, 45 tokens/s
output).

| File | Content |
|------|---------|
| `__GGUF__` | current model, SHA-256 `__SHA256__` |
| `latest.json` | manifest of the current model, used by the add-on for automatic updates |
| `chat_template.jinja` | chat template embedded in the GGUF |

## Usage

```bash
llama-server -m __GGUF__ --jinja -c 8192 --port 8080
```

`--jinja` is required: it activates the embedded chat template, which renders
the tools and parses `<tool_call>` blocks into OpenAI `tool_calls`.
Recommended sampling: temperature 0.

The model expects the system prompt and the two tools from
[`llm_schema.py`](https://huggingface.co/datasets/__DATASET_REPO__/blob/main/llm_schema.py)
(prompt schema 5): a German instruction, the list of exposed entities
(`entity_id | name | area`), `control` (calls Home Assistant services, for
lights also by `area`, `"*"` for the whole house) and `get_state`. The Home
Assistant integration and add-on of the project repository build this prompt
automatically; the integration resolves areas to entity_ids.

### Example system prompt

```text
Du bist der Sprachassistent eines Smart Homes mit Home Assistant. Antworte immer auf Deutsch, kurz und freundlich.
Steuere Geräte mit dem Werkzeug control. Frage Zustände und Messwerte mit get_state ab, bevor du Fragen dazu beantwortest.
Verwende nur entity_ids aus der folgenden Liste. Gibt es ein Gerät nicht, sage das, statt eine entity_id zu erfinden.
Meint der Nutzer alle Lichter eines Bereichs, gib statt entity_id den Bereich an (area), für das ganze Haus area "*".

Geräte (entity_id | Name | Bereich):
light.lumi_lumi_sensor_magnet_aq2_00158dff | Wandlampe Carport | Carport
sensor.carport_leistung | Carport Leistung | Carport
switch.carport_bewaesserung | Bewässerung Carport | Carport
valve.carport_bewaesserung | Wasserventil | Carport
climate.thermostat_gaestezimmer | Thermostat Gästezimmer | Gästezimmer
vacuum.gaestezimmer_saugroboter | Saugroboter | Gästezimmer
sensor.waschraum_temperatur | Temperatur Waschraum | Waschraum
sensor.waschraum_waschmaschine | Waschmaschine | Waschraum
switch.waschraum_entfeuchter | Waschraum Luftentfeuchter | Waschraum
switch.waschraum_waschmaschine_power | Waschmaschine Steckdose | Waschraum
alarm_control_panel.haus_alarmanlage | Alarmanlage | -
binary_sensor.haus_wohnungstuer_state | Wohnungstüre | -
button.haus_bewaesserung_garten | Gartenbewässerung | -
lock.haus_terrassentuer | Terrassentür | -
lock.haustuer_haus | Haustür | -
scene.haus_hell | Hell | -
scene.haus_konzentration | Konzentration | -
script.haus_kaffee_kochen | Kaffee kochen | -
```

If the client sends no system message, the embedded template uses the first
line of this prompt. Without the device list the model cannot control
anything, so always send the full prompt.

### Tool results

`control` returns `{"success": true}`, with an area also the switched lights
in `done`. If a device fails it returns
`{"success": false, "error": "...", "done": [...]}`, and the model reports
the failure instead of claiming success. `get_state` returns a list with
`entity_id`, `name`, `state` and attributes such as `unit_of_measurement`,
`device_class`, `brightness_pct` or `current_temperature`.

### Full chat in the model's format

Two turns: a light switched by area and a state question about an
unreachable lock. This is the exact text the model sees, including the
special tokens `<|im_start|>` and `<|im_end|>` and the `<tool_call>` and
`<tool_response>` blocks (rendered with the embedded template and
`llm_schema.TOOLS`). The model generates everything after
`<|im_start|>assistant` up to and including `<|im_end|>`; tool results come
back as a user turn with `<tool_response>`.

```text
<|im_start|>system
Du bist der Sprachassistent eines Smart Homes mit Home Assistant. Antworte immer auf Deutsch, kurz und freundlich.
Steuere Geräte mit dem Werkzeug control. Frage Zustände und Messwerte mit get_state ab, bevor du Fragen dazu beantwortest.
Verwende nur entity_ids aus der folgenden Liste. Gibt es ein Gerät nicht, sage das, statt eine entity_id zu erfinden.
Meint der Nutzer alle Lichter eines Bereichs, gib statt entity_id den Bereich an (area), für das ganze Haus area "*".

Geräte (entity_id | Name | Bereich):
light.lumi_lumi_sensor_magnet_aq2_00158dff | Wandlampe Carport | Carport
sensor.carport_leistung | Carport Leistung | Carport
switch.carport_bewaesserung | Bewässerung Carport | Carport
valve.carport_bewaesserung | Wasserventil | Carport
climate.thermostat_gaestezimmer | Thermostat Gästezimmer | Gästezimmer
vacuum.gaestezimmer_saugroboter | Saugroboter | Gästezimmer
sensor.waschraum_temperatur | Temperatur Waschraum | Waschraum
sensor.waschraum_waschmaschine | Waschmaschine | Waschraum
switch.waschraum_entfeuchter | Waschraum Luftentfeuchter | Waschraum
switch.waschraum_waschmaschine_power | Waschmaschine Steckdose | Waschraum
alarm_control_panel.haus_alarmanlage | Alarmanlage | -
binary_sensor.haus_wohnungstuer_state | Wohnungstüre | -
button.haus_bewaesserung_garten | Gartenbewässerung | -
lock.haus_terrassentuer | Terrassentür | -
lock.haustuer_haus | Haustür | -
scene.haus_hell | Hell | -
scene.haus_konzentration | Konzentration | -
script.haus_kaffee_kochen | Kaffee kochen | -

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "control", "description": "Steuert ein oder mehrere Geräte. Jede Aktion ruft einen Home-Assistant-Dienst für eine oder mehrere entity_ids auf, bei Lichtern auch für alle Lichter eines Bereichs (area). Erlaubte Dienste: light.turn_on, light.turn_off, switch.turn_on, switch.turn_off, input_boolean.turn_on, input_boolean.turn_off, fan.turn_on, fan.turn_off, fan.set_percentage, cover.open_cover, cover.close_cover, cover.stop_cover, cover.set_cover_position, climate.set_temperature, climate.turn_on, climate.turn_off, lock.lock, lock.unlock, alarm_control_panel.alarm_arm_home, alarm_control_panel.alarm_arm_away, alarm_control_panel.alarm_arm_night, alarm_control_panel.alarm_disarm, media_player.turn_on, media_player.turn_off, media_player.media_play, media_player.media_pause, media_player.media_stop, media_player.media_next_track, media_player.media_previous_track, media_player.volume_set, media_player.volume_up, media_player.volume_down, vacuum.start, vacuum.pause, vacuum.return_to_base, scene.turn_on, script.turn_on, button.press, lawn_mower.start_mowing, lawn_mower.pause, lawn_mower.dock, valve.open_valve, valve.close_valve.", "parameters": {"type": "object", "properties": {"actions": {"type": "array", "items": {"type": "object", "properties": {"service": {"type": "string", "description": "Dienst im Format domain.dienst"}, "entity_id": {"type": "array", "items": {"type": "string"}}, "area": {"type": "string", "description": "Statt entity_id: alle Lichter dieses Bereichs, \"*\" für das ganze Haus"}, "data": {"type": "object", "description": "Optionale Parameter, z. B. brightness_pct, brightness_step_pct, color_name, color_temp_kelvin, position, temperature, percentage, volume_level, code"}}, "required": ["service"]}}}, "required": ["actions"]}}}
{"type": "function", "function": {"name": "get_state", "description": "Liefert den aktuellen Zustand und wichtige Attribute von Geräten und Sensoren, entweder für entity_ids oder für alle Lichter eines Bereichs (domain light und area).", "parameters": {"type": "object", "properties": {"entity_ids": {"type": "array", "items": {"type": "string"}}, "domain": {"type": "string"}, "area": {"type": "string"}}}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
<|im_start|>user
Setz das Licht im Carport auf 75 Prozent Helligkeit<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "control", "arguments": {"actions": [{"service": "light.turn_on", "area": "Carport", "data": {"brightness_pct": 75}}]}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"success": true, "done": ["light.lumi_lumi_sensor_magnet_aq2_00158dff"]}
</tool_response><|im_end|>
<|im_start|>assistant
Das Licht im Carport ist jetzt auf 75 Prozent.<|im_end|>
<|im_start|>user
Was macht Terrassentür?<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_state", "arguments": {"entity_ids": ["lock.haus_terrassentuer"]}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
[{"entity_id": "lock.haus_terrassentuer", "name": "Terrassentür", "state": "unavailable"}]
</tool_response><|im_end|>
<|im_start|>assistant
Das kann ich nicht sagen: Terrassentür ist gerade nicht erreichbar.<|im_end|>

```

Via the OpenAI-compatible API of `llama-server` the same conversation is
sent as messages, and llama.cpp renders the template:

```json
{
  "model": "ha-llm",
  "temperature": 0,
  "tools": ["... llm_schema.TOOLS ..."],
  "messages": [
    {"role": "system", "content": "... system prompt above ..."},
    {"role": "user", "content": "Setz das Licht im Carport auf 75 Prozent Helligkeit"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
      "function": {"name": "control", "arguments": "{\"actions\": [{\"service\": \"light.turn_on\", \"area\": \"Carport\", \"data\": {\"brightness_pct\": 75}}]}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"success\": true, \"done\": [\"light.lumi_lumi_sensor_magnet_aq2_00158dff\"]}"},
    {"role": "assistant", "content": "Das Licht im Carport ist jetzt auf 75 Prozent."},
    {"role": "user", "content": "Was macht Terrassentür?"}
  ]
}
```

## Automatic updates

`latest.json` describes the current model:

```json
__MANIFEST__
```

The add-on *HA LLM Inference* (0.8.0 or later) reads it at start and every
6 hours if `auto_update` is on (default) and no own model is configured. It
downloads a new file while the server keeps running, checks the SHA-256,
switches over and deletes the old file. A model with a higher `schema` than
the add-on supports is skipped until the add-on is updated.

## Training

- Data: [__DATASET_REPO__](https://huggingface.co/datasets/__DATASET_REPO__),
  8,000 synthetic German conversations over randomly generated houses
  (up to 110 entities, group entities, rooms without lights, lights named
  after their room, several irrigation devices, brand names, devices that are
  not in the list, unreachable devices, rooms where only some lights are
  reachable, users insisting on missing devices, heating mode switches,
  door contacts, electric cars, outdoor temperature next to forecast and
  weather sensors, area targeting, multi-turn conversations). Each of the 68
  generator cases has the same quota, rare cases included.
- Model soup: average of the LoRA weight deltas of three runs (seeds 13, 14,
  15, generator states of increasing coverage). Each run: LoRA r=32,
  alpha=64, all linear layers, 2.5 epochs from Qwen2.5-3B-Instruct, lr 2e-4,
  loss on assistant tokens only, RTX 3090, 5.5 hours. Single runs reach
  similar test scores but make different individual mistakes; the average
  keeps what they agree on.
- Quantization: Q4_K_M with an importance matrix computed from 300 training
  conversations. Without it, Q4_K_M broke single runs (291/300 in f16,
  243/300 in Q4_K_M).
- The embedded chat template is the Qwen2.5 template with a German default
  system prompt. For all 8,300 training and test conversations it renders
  the same text as the template used in training.

## Honest answers

The training data teaches the model to report failed devices instead of
claiming success, including in later turns ("Ist es jetzt an?", "Versuch es
nochmal"). Every training answer is checked against the preceding tool
results; no answer claims a success the tool did not report.

## Evaluation

- 300 held-out conversations in houses not seen during training: 99.7 %
  correct tool calls, 100 % grounded answers, 17/17 missing devices refused,
  71/72 state questions.
- A real house with over 100 exposed entities: 26/26 and 32/32 sentences on
  two snapshots of the device list, 128/128 on four variants with 8 devices
  removed each, 30/30 questions about missing devices.
- 19 simulated multi-turn dialogues with unreachable devices: no claimed
  success for a failed device. With two of four kitchen lights unreachable,
  "Mach das Licht in der Küche an" switches the room by area and names the
  switched and the unreachable lights.
- End-to-end in Home Assistant 2026.10.4: 24/24 test sentences, median
  latency 2.9 s on 8 CPU threads.

Known weaknesses: device names are not always reproduced in the exact
spelling of the list (a space inside a compound name), and a status answer
about a room can leave out a switched-off lamp named like the room.

## Limitations

- German only; supports the services listed in `llm_schema.SERVICES`.
- No timers, automations or general knowledge questions. Weather forecasts
  only through exposed forecast sensors.
- Above ~250 exposed entities the 8k context is not sufficient.

## License

Derivative of Qwen2.5-3B-Instruct, distributed under the
[Qwen RESEARCH LICENSE AGREEMENT](LICENSE): **non-commercial use only**.
Modifications: LoRA fine-tuning (merged; the file averages three LoRA runs),
conversion to GGUF, Q4_K_M quantization with an importance matrix and a
German default system prompt in the chat template. See `NOTICE`.
