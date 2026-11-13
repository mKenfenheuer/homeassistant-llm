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
| German | `de/` | v17 |

Models for English, Spanish, French, Italian and Dutch are trained from the
same generator (see the dataset) and are added to this repository as they
pass their tests.

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
llama-server -m de/qwen2.5-3b-ha-v17-Q4_K_M.gguf --jinja -c 8192 --port 8080
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
      "version": "17",
      "file": "de/qwen2.5-3b-ha-v17-Q4_K_M.gguf",
      "sha256": "...",
      "size": 1929902976,
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

## Training (German v17)

- Data: [__DATASET_REPO__](https://huggingface.co/datasets/__DATASET_REPO__)
  (rows with `locale` = `de`): 27,140 synthetic conversations over randomly
  generated houses, up to 236 per generator case for each of the 115 cases
  (rare cases get the same share). Besides device control and state
  questions: music and the media library, the speaker of the voice device,
  seeking and resuming, timers, lists, weather reports per answer length,
  time, broadcasts, unreachable devices, missing devices and multi-turn
  conversations. The answer length of each conversation is sampled (short
  20 %, medium 40 %, long 20 %, detailed 20 %).
- Continued from the LoRA adapter of v16 (r=32, alpha=64, all linear layers,
  Qwen2.5-3B-Instruct), lr 1e-4, loss on assistant tokens only, RTX 3090. The
  eval loss on 99 held-out conversations was measured every 10 steps; the
  run stopped at step 730 of 1,675 when it had not dropped by 2 % for 20
  measurements, after about 5 hours. The released file is the step with the lowest
  eval loss (540, 0.0246).
- Quantization: Q4_K_M with an importance matrix computed from 300 training
  conversations.

## Honest answers

The training data teaches the model to report failed devices instead of
claiming success, including in later turns. Every training answer is checked
against the preceding tool results; no answer claims a success the tool did
not report.

## Evaluation (German v17)

- 300 held-out conversations in houses not seen during training: 97.7 %
  correct tool calls (v16 on the same set: 94.7 %), 97.2 % grounded answers.
- Acceptance test on a real house (over 100 exposed entities, 55 scored
  turns including the media library, covers and windows of the whole house
  and two questions in one sentence): 53/55, v16 45/55.
- End to end in Home Assistant 2026.12 (dev): media library, search, resume
  and seeking on the speaker of a voice satellite.

Known weaknesses: in a media library with many playlists (76 in the test
house) "Spiel meine Lieblingssongs" or "Spiel eine Playlist zum Einschlafen"
can miss the matching playlist and name a few others instead. Listing many
open covers or lights can leave one out.

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
