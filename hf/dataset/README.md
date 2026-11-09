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
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: test
        path: eval.jsonl
---

# Home Assistant Tool Calling (German, synthetic)

German conversations between a user and a smart-home assistant that controls
Home Assistant with two tools: `control` (Home Assistant services such as
`light.turn_on` or `cover.close_cover`) and `get_state`.

Every sample uses a freshly generated house: random rooms, German device
names and partly cryptic entity_ids like `light.shelly1_3c61a5`. A model
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
- **Model**: [mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF](https://huggingface.co/mkenfenheuer/qwen2.5-3b-homeassistant-de-GGUF),
  Qwen2.5-3B-Instruct fine-tuned on this dataset.

| Split | Samples | Seed |
|-------|---------|------|
| train | 8,000 | 15 |
| test | 300 | 1015 (different houses) |

## Format

One JSON object per line, OpenAI chat format. Tool call arguments are JSON
strings. `meta` holds the category and the gold tool calls for evaluation.

```json
{"messages": [
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

## Categories (train)

| Category | Samples | Content |
|----------|---------|---------|
| command | 2,850 | single commands: lights (also by area), covers, awning, climate, locks, alarm, media, fans, vacuum, lawn mower, valves, buttons, scenes, group entities, appliance power switches, heating mode switches |
| question | 1,917 | state questions answered via `get_state`: appliance status sensors, door contacts, electric cars, outdoor temperature next to forecast and weather sensors, lights of an area |
| conversation | 1,641 | 3 to 6 turns: undo, corrections, "Welches Licht hast du ausgeschaltet?", "Welche Heizungen hast du verstellt?", recaps after a correction, checks, topic changes |
| error | 546 | unreachable devices, partial success, rooms where only some lights are reachable, retries: the answer reports the failure |
| unknown | 501 | devices or rooms that do not exist, or a room without that device type: no tool call, other rooms suggested, also when the user insists twice; houses without areas |
| followup | 198 | two-turn conversations ("und wieder aus", "Nee, ist es nicht", accepting a suggested other room) |
| compound | 118 | two commands in one sentence |
| relative | 118 | "Mach es im Bad wärmer": read state, then set |
| chat | 111 | greetings, thanks, capabilities |

The generator draws from 68 cases with equal quotas (111 to 118 samples each),
so rare situations are as frequent as common ones.

`src/check_dataset.py` verifies every assistant answer against the preceding
tool results (no claimed success after an error, numbers present in the
result, unreachable devices mentioned).

About 20 % of the samples start with an earlier, unrelated turn, so the model
learns to act on the latest request only.

## Generation

Created with `src/gen_dataset.py` (template-based, no LLM involved):
`python src/gen_dataset.py data/train.jsonl 8000 15`.

The current model averages three runs trained on sets from seeds 13, 14 and
15 with successive generator versions; this repository holds the set of the
last run (seed 15), which the published generator reproduces. The vendor id
prefixes in the generator (Homematic, Shelly) are fictional.
