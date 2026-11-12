---
license: cc-by-4.0
language:
  - de
task_categories:
  - text-generation
tags:
  - home-assistant
  - tool-calling
  - function-calling
  - smart-home
  - synthetic
size_categories:
  - 10K<n<100K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: test
        path: eval.jsonl
---

# Home Assistant Tool Calling (synthetic)

Conversations between a user and a smart-home assistant that controls
Home Assistant with five tools: `control` (Home Assistant services such as
`light.turn_on`, `cover.close_cover`, music search and to-do lists),
`get_state` (states, weather with the course of the day and forecast, list
items), `timer` (timers of the voice device, also delayed commands),
`get_time` and `broadcast`. Together they cover the built-in Assist intents of
Home Assistant.

Every language has its own model, so every row carries a `locale` column
(`de`, `en`, `es`, `fr`, `it`, `nl`). Filter by it to train a model for one language:

```python
from datasets import load_dataset
de = load_dataset("mkenfenheuer/homeassistant-tool-calling", split="train").filter(
    lambda r: r["locale"] == "de")
```

Every sample uses a freshly generated house: random rooms, device names in
the language of the sample and partly cryptic entity_ids like `light.shelly1_3c61a5`. A model
trained on it learns to resolve names via the device list in the system prompt
instead of memorising one house.

## Project

The dataset belongs to [homeassistant-llm](https://github.com/mKenfenheuer/homeassistant-llm). That repository holds the
generator, the training scripts and everything needed to run the model in
Home Assistant:

- **Add-on** *HA LLM Inference*: runs `llama-server` with the model on the
  Home Assistant host and keeps the model up to date. Add `https://github.com/mKenfenheuer/homeassistant-llm` under
  Settings → Add-ons → Add-on Store → ⋮ → Repositories.
- **Integration** *HA LLM Agent (lokal)*: a conversation agent for Assist that
  builds the system prompt from the exposed entities and executes the tool
  calls. Install it via HACS as a custom repository (type Integration).
- **Model**: [mkenfenheuer/qwen2.5-3b-homeassistant-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-GGUF),
  Qwen2.5-3B-Instruct fine-tuned on this dataset.

__SPLITS__

The test split uses other houses (seed + 1000) than the train split.

## Format

One JSON object per line, OpenAI chat format, with the `locale` of the sample. Tool call arguments are JSON
strings. `meta` holds the category and the gold tool calls for evaluation.

```json
{"locale": "de", "messages": [
  {"role": "system", "content": "Du bist der Sprachassistent ... Geräte (entity_id | Name | Bereich):\nlight.kueche_decke | Deckenlicht Küche | Küche\n..."},
  {"role": "user", "content": "Mach das Licht in der Küche an"},
  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_00001", "type": "function",
    "function": {"name": "control", "arguments": "{\"actions\": [{\"service\": \"light.turn_on\", \"entity_id\": [\"light.kueche_decke\"]}]}"}}]},
  {"role": "tool", "tool_call_id": "call_00001", "content": "{\"success\": true}"},
  {"role": "assistant", "content": "Das Licht in der Küche ist jetzt an."}],
 "meta": {"cat": "command", "gold": [["control", {"actions": [...]}]], "location": null}}
```

The tool schema (`TOOLS`) is not part of each row; it is defined in
`llm_schema.py` of the [project repository](https://github.com/mKenfenheuer/homeassistant-llm) and passed to the chat template
during training.

## Categories (train, de)

| Category | Samples | Content |
|----------|---------|---------|
| command | 7,903 | single commands: lights (also by area), covers, awning, climate, humidifiers, locks, alarm, media, music search and play (artist, track, album, playlist, radio), shuffle, repeat, mute, to-do lists (add, check off, remove), fans, vacuum, lawn mower, valves with position, buttons, scenes, group entities, appliance power switches, heating mode switches |
| question | 5,964 | state questions answered via `get_state`, `get_time` or `timer`: weather now, for the rest of the day and the next days, to-do list items, time and date, timers still running, appliance status sensors, door contacts, electric cars, outdoor temperature, lights of an area |
| conversation | 4,640 | 3 to 6 turns: undo, corrections, "Welches Licht hast du ausgeschaltet?", recaps, checks, topic changes, music follow-ups ("Lauter", "Nächstes Lied"), timer follow-ups ("Wie lange noch?"), list follow-ups ("Und Eier"), choosing a speaker after a question back |
| timer | 2,289 | start (also named and with a delayed command such as "Mach in 10 Minuten das Licht aus"), cancel, cancel all, pause, resume, add, subtract, ambiguous names, no matching timer, devices without timer support |
| unknown | 1,502 | devices, rooms or lists that do not exist, alarms at a clock time: no tool call, other rooms suggested, also when the user insists twice; houses without areas |
| error | 1,128 | unreachable devices, partial success, rooms where only some lights are reachable, retries, music not found: the answer reports the failure |
| broadcast | 488 | announcements to all voice devices, also without other devices |
| followup | 424 | two-turn conversations ("und wieder aus", "Nee, ist es nicht", accepting a suggested other room) |
| clarify | 245 | several speakers and none named: the assistant asks which one |
| compound | 245 | two commands in one sentence |
| relative | 244 | "Mach es im Bad wärmer": read state, then set |
| chat | 180 | greetings, thanks, capabilities, "Vergiss es" |

The generator draws from 107 cases with equal quotas (236 samples each, a few
cases with few distinct sentences have fewer), so rare situations are as
frequent as common ones.

## Answer length

The system prompt carries one of four answer lengths, set in the integration
options: `short` ("Antworte so knapp wie möglich"), `medium` (default),
`long` and `detailed`. Answers follow it: a weather question gets the current
weather only (short), the course of the rest of the day (medium), today's
range and rain chance (long), and humidity, wind and tomorrow (detailed).
`meta.verbosity` holds the level of each sample (40 % medium, 20 % each other
level).

`src/check_dataset.py` verifies every assistant answer against the preceding
tool results (no claimed success after an error, numbers present in the
result, unreachable devices mentioned).

About 20 % of the samples start with an earlier, unrelated turn, so the model
learns to act on the latest request only.

## Generation

Created with `gen_dataset.py` and the German generator package `gen/de/`
(template-based, no LLM involved):
`python gen_dataset.py --lang de data/de/train.jsonl per_case:236 16`.
Every language has its own generator package and dataset; `llm_schema.py`
holds the prompt texts of all languages.

The vendor id prefixes in the generator (Homematic, Shelly) are fictional.
