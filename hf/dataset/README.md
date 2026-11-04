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

| Split | Samples | Seed |
|-------|---------|------|
| train | 6,000 | 1 |
| test | 300 | 999 (different houses) |

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
`llm_schema.py` of the project repository and passed to the chat template
during training.

## Categories (train)

| Category | Samples | Content |
|----------|---------|---------|
| command | 3,088 | single commands: lights, covers, climate, locks, alarm, media, fans, vacuum, scenes |
| question | 1,458 | state questions answered via `get_state` |
| followup | 560 | two-turn conversations ("und wieder aus") |
| compound | 392 | two commands in one sentence |
| unknown | 252 | devices or rooms that do not exist, no tool call |
| error | 120 | failed service call reported to the user |
| relative | 95 | "Mach es im Bad wärmer": read state, then set |
| chat | 35 | greetings, thanks, capabilities |

## Generation

Created with `src/gen_dataset.py` (template-based, no LLM involved):
`python src/gen_dataset.py data/train.jsonl 6000 1`.
