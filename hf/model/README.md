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
datasets:
  - __DATASET_REPO__
---

# Qwen2.5-3B Home Assistant (GGUF)

**Built with Qwen.** LoRA fine-tunes of
[Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
that control a Home Assistant installation via tool calls: devices, music
and the media library of a player, timers, shopping lists, weather, time and
broadcasts. One model per language,
each quantized to Q4_K_M (1.9 GB) with an importance matrix. Runs on a CPU
with AVX2 (about 16 tokens/s output on 4 cores) or on small GPUs (GTX 1060
6 GB: 540 tokens/s prompt, 45 tokens/s output).

| Language | Folder | Version |
|----------|--------|---------|
| German | `de/` | v19 |
| English | `en/` | v19 |
| Spanish | `es/` | v19 |
| French | `fr/` | v19 |
| Italian | `it/` | v18 |
| Dutch | `nl/` | v19 |

All six languages are trained from the same generator (see the dataset).

| File | Content |
|------|---------|
| `models.json` | language → current model (file, SHA-256, size, prompt schema); read by the add-on for automatic updates |
| `<lang>/qwen2.5-3b-ha-v<version>-Q4_K_M.gguf` | current model of the language |
| `<lang>/chat_template.jinja` | chat template embedded in the GGUF, with the default system prompt of the language |

The root folder still holds `latest.json` and the German v15.1 GGUF for
add-ons older than 0.13.0.

## Home Assistant

The project repository [homeassistant-llm](https://github.com/mKenfenheuer/homeassistant-llm)
contains the add-on *HA LLM Inference*, which runs the model on the Home
Assistant host and updates it automatically, and the integration *HA LLM
Agent (local)*, a conversation agent for Assist. Add the repository as an
add-on repository and as a custom HACS repository (type Integration). A
server with a GPU works as well: the integration only needs the URL of a
`llama-server`. In the integration options you choose the language and the
answer length (short, medium, long, detailed).

## Usage

```bash
llama-server -m de/qwen2.5-3b-ha-v19-Q4_K_M.gguf --jinja -c 8192 --port 8080
```

`--jinja` is required: it activates the embedded chat template, which renders
the tools and parses `<tool_call>` blocks into OpenAI `tool_calls`.
Recommended sampling: temperature 0.

The model expects the system prompt and the six tools from
[`llm_schema.py`](https://huggingface.co/datasets/__DATASET_REPO__/blob/main/llm_schema.py)
(prompt schema 7):

| Tool | Purpose |
|------|---------|
| `control` | calls Home Assistant services for entity_ids, for lights also by `area` (`"*"` = whole house); also `media_player.search_and_play` (query, media_type, artist; without query the loaded title goes on, else the favorite songs), `media_player.seek_relative` (seconds) and `todo.add_item` / `todo.complete_item` |
| `get_state` | states and attributes of entity_ids or of all devices of a domain in an area (`"*"` = whole house, optional `device_class` such as `window`); weather with the hours left today (`later_today`) and the next days (`forecast`), to-do lists with their `items` |
| `music_library` | names in the media library of a player: playlists, artists, albums, tracks, stations, podcasts, optionally filtered by `search` |
| `timer` | start, cancel, pause, resume, add, subtract, status of timers on the voice device; optional delayed command |
| `get_time` | current time and date |
| `broadcast` | announcement on all speakers |

The integration builds the prompt and executes the tools with the built-in
intents of Home Assistant (timers, broadcast) and the media player services
(`browse_media`, `search_media`, `play_media`, `media_seek`).

### Example system prompt (German, answer length medium)

```text
Du bist der Sprachassistent eines Smart Homes mit Home Assistant. Antworte immer auf Deutsch und freundlich. Antworte kurz, meist in einem Satz.
Steuere Geräte mit dem Werkzeug control. Frage Zustände und Messwerte mit get_state ab, bevor du Fragen dazu beantwortest.
Verwende nur entity_ids aus der folgenden Liste. Gibt es ein Gerät nicht, sage das, statt eine entity_id zu erfinden.
Meint der Nutzer alle Lichter eines Bereichs, gib statt entity_id den Bereich an (area), für das ganze Haus area "*".
Timer stellst du mit timer, Uhrzeit und Datum liefert get_time, Durchsagen an alle Lautsprecher machst du mit broadcast.

Geräte (entity_id | Name | Bereich):
cover.rolladen_kueche | Rollladen Küche | Küche
media_player.kueche | Lautsprecher Küche | Küche
light.wohnzimmer | Wohnzimmer | Wohnzimmer
todo.einkaufsliste | Einkaufsliste | -
weather.zuhause | Zuhause | -

Das Sprachgerät steht im Bereich: Küche
Der Lautsprecher des Sprachgeräts: media_player.kueche
```

The sentence on answer length comes from `LANGUAGES[lang]["verbosity"]`, the
last two lines from the area of the voice device and its own media player
(music without a named player plays there). If the client sends no system
message, the embedded template uses a short default prompt; without the device
list the model cannot control anything, so always send the full prompt.

### Tool results

`control` returns `{"success": true}`, with an area also the switched
entities in `done`. If a device fails it returns
`{"success": false, "error": "...", "done": [...]}`, and the model reports
the failure instead of claiming success. `timer` returns the started or
matched timers or an error such as `timers are not supported on this device`.

### Example via the OpenAI-compatible API

```json
{
  "model": "ha-llm",
  "temperature": 0,
  "tools": ["... llm_schema.build_tools('de') ..."],
  "messages": [
    {"role": "system", "content": "... system prompt above ..."},
    {"role": "user", "content": "Spiel Musik von Queen in der Küche"},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function",
      "function": {"name": "control", "arguments": "{\"actions\": [{\"service\": \"media_player.search_and_play\", \"entity_id\": [\"media_player.kueche\"], \"data\": {\"query\": \"Queen\", \"media_type\": \"artist\"}}]}"}}]},
    {"role": "tool", "tool_call_id": "call_1", "content": "{\"success\": true}"},
    {"role": "assistant", "content": "Ich spiele Musik von Queen auf Lautsprecher Küche."},
    {"role": "user", "content": "Stell einen Timer auf 10 Minuten für die Pizza"}
  ]
}
```

## Automatic updates

`models.json` lists the current model of every language:

```json
{
  "languages": {
    "de": {
      "version": "19",
      "file": "de/qwen2.5-3b-ha-v19-Q4_K_M.gguf",
      "sha256": "...",
      "size": 1929902880,
      "schema": 7,
      "min_addon_version": "0.14.0"
    }
  }
}
```

The add-on *HA LLM Inference* (0.13.0 or later) reads it at start and every
6 hours if `auto_update` is on (default) and no own model is configured. It
downloads a new file while the server keeps running, checks the SHA-256,
switches over and deletes the old file. A model with a higher `schema` than
the add-on supports is skipped until the add-on is updated.

## Training (v19: German, English, Spanish, French, Dutch)

- Data: [__DATASET_REPO__](https://huggingface.co/datasets/__DATASET_REPO__),
  one generator per language, up to 236 conversations for each of 120 cases
  (rare cases get the same share), about 28,000 per language. Besides device
  control and state questions: music and the media library, the speaker of
  the voice device, timers, lists, weather reports per answer length, time,
  broadcasts, unreachable devices, missing devices and multi-turn
  conversations. The answer length of each conversation is sampled (short
  20 %, medium 40 %, long 20 %, detailed 20 %).
- New in v19, from the failures of v18 in the acceptance test: lamps named
  after the fixture they light and groups named after a floor ("the light
  over the dining table", "the shutters upstairs"), whose words are part of
  a device name and not an area; the Do-not-disturb and Mute switches of
  Echo devices; battery sensors of mowers and vacuums; a radiator and a split
  unit in one room ("the heating" is the radiator); temperature sensors in
  outdoor areas count as outdoor sensors. The fixtures and the ground floor
  of the public acceptance house are kept out of the training data.
- Each language continued from its own v18 LoRA adapter (r=32, alpha=64, all
  linear layers, Qwen2.5-3B-Instruct) with lr 1e-4 in
  [AI Studio](https://github.com/mKenfenheuer/ai-studio) on an RTX 3090,
  loss on assistant tokens only, eval loss on about 100 held-out
  conversations every 20 steps, stop after 10 checks without improvement.
  Released steps (eval loss): German 500 (0.0281), English 700 (0.0290),
  Spanish 300 (0.0228), French 460 (0.0248), Dutch 180 (0.0255).
- Quantization: Q4_K_M with an importance matrix from 300 training
  conversations of the language, rendered with the model's chat template and
  tools.

## Training (Italian v18)

The Italian v19 run did not beat v18 in the acceptance test (39/41
against 40/41), so Italian stays on v18.


- Data: the rows with `locale` = `it` of the same dataset, 26,921
  conversations of the Italian generator.
- Continued from the German v18 adapter with lr 1e-4 in AI Studio on an
  RTX 3090, early stop on the eval loss of 100 held-out conversations.
  Released step: 1220, eval loss 0.0266.
- Quantization: Q4_K_M with an importance matrix from 300 Italian training
  conversations.

## Honest answers

The training data teaches the model to report failed devices instead of
claiming success, including in later turns. Every training answer is checked
against the preceding tool results; no answer claims a success the tool did
not report.

## Evaluation (v19)

300 held-out conversations per language in houses not seen during training,
from the v19 generators; v18 was measured on the same conversations.

| Language | Tool calls v19 | Tool calls v18 | Grounded answers v19 | Grounded answers v18 |
|----------|----------------|----------------|----------------------|----------------------|
| German | 98.3 % | 96.7 % | 97.2 % (69/71) | 95.8 % (68/71) |
| English | 98.0 % | 96.3 % | 95.8 % (69/72) | 97.2 % (69/71) |
| Spanish | 99.0 % | 97.3 % | 98.6 % (72/73) | 98.6 % (72/73) |
| French | 99.7 % | 98.0 % | 97.3 % (72/74) | 97.3 % (71/73) |
| Dutch | 97.3 % | 97.0 % | 98.6 % (69/70) | 98.6 % (70/71) |
| Italian | 97.0 % | 97.3 % | 98.6 % (72/73) | 98.6 % (71/72) |

German was also tested on a real house (over 100 exposed entities, 36
scored turns): v19 32/36, v18 28/36. v19 no longer finds the air conditioner
of the living room when asked for "die Klimaanlage", which v18 did.

Weakest categories in v19: compound questions in English (4/6) and German
(5/6), where one of two devices is left out; state questions in Dutch
(73/77).

## Evaluation (Italian v18)

- 300 held-out conversations of the Italian generator in houses not seen
  during training: 99.3 % correct tool calls, 100 % grounded answers
  (69/69), 0.66 s per conversation on an RTX 3090 (shared with an export
  running at the same time). Both misses read a thermostat or the outdoor
  sensor instead of the room's temperature sensor.

## Acceptance test (public)

A house built like a real installation -- 107 entities in 15 areas, cryptic
entity ids (Homematic, Shelly, Tasmota), groups, ten offline devices,
plants, a car, a mower, a 3D printer and a weather station -- in which every
device and area name is new to the training data of the language. 40
scenarios (41 scored turns): switching by name and by room, groups, shutter
positions, a follow-up ("dim it to 30 percent"), offline devices, state
questions, a missing device, timers, broadcasts, time and small talk. The
house, the names and the sentences are in the GitHub repository
([tests/acceptance](https://github.com/mKenfenheuer/homeassistant-llm/tree/main/tests/acceptance));
run it with `src/agent_sim.py`.

| Model | Passed | Failed turn |
|-------|--------|-------------|
| German v19 | 41/41 | (v18: 40/41, "Mach das Licht am Waschtischschrank an" took the device for an area) |
| English v19 | 41/41 | (v18: 40/41, "What's the temperature outside?" read the car's charge level) |
| Spanish v19 | 41/41 | (v18: 38/41, fixture lamp, Do not disturb, mower battery) |
| French v19 | 39/41 | "le radiateur de la salle commune" set the split unit; "la menthe" read the lemon tree's soil moisture (v18: 38/41) |
| Italian v18 | 40/41 | "le tapparelle del pianterreno" was taken for a room (v19: 39/41, not released) |
| Dutch v19 | 41/41 | (v18: 40/41, "Doe het licht bij de wastafelkast aan" took the device for a room) |

No answer of any model contained a word of another language.

## Limitations

- Supports the services listed in `llm_schema.SERVICES`.
- No automations or general knowledge questions.
- Above ~250 exposed entities the 8k context is not sufficient.

## License

Derivative of Qwen2.5-3B-Instruct, distributed under the
[Qwen RESEARCH LICENSE AGREEMENT](LICENSE): **non-commercial use only**.
Modifications: LoRA fine-tuning (merged), conversion to GGUF, Q4_K_M
quantization with an importance matrix and a default system prompt per
language in the chat template. See `NOTICE`.
