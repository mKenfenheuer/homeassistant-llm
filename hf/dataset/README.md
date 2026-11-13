---
license: cc-by-4.0
language:
  - de
  - en
  - es
  - fr
  - it
  - nl
task_categories:
  - text-generation
tags:
  - home-assistant
  - tool-calling
  - function-calling
  - smart-home
  - synthetic
size_categories:
  - 100K<n<1M
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
Home Assistant with six tools: `control` (Home Assistant services such as
`light.turn_on`, `cover.close_cover`, music search and playback, seeking,
to-do lists), `get_state` (states of entities or of all devices of a domain in
an area, weather with the course of the day and forecast, list items),
`music_library` (playlists, artists, albums, tracks, stations and podcasts in
the media library of a player), `timer` (timers of the voice device, also
delayed commands), `get_time` and `broadcast`. Together they cover the
built-in Assist intents of Home Assistant.

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
- **Integration** *HA LLM Agent (local)*: a conversation agent for Assist that
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

## Categories (train, de, v18)

| Category | Samples | Content |
|----------|---------|---------|
| command | 9,209 | single commands: lights (also by area), covers, awning, climate, humidifiers, locks, alarm, media (play, pause, next and previous track, volume, seeking forward and back), music search and play (artist, track, album, playlist, radio, plain "Spiel Musik"), playlists picked from the media library by mood or as favorite songs, the speaker of the voice device preferred, shuffle, repeat, mute, to-do lists (add, check off, remove), fans, vacuum, lawn mower, valves with position, buttons, scenes, group entities, appliance power switches, heating mode switches; short media commands without a named player read all players and act on the one that plays (or, for "Play", is paused) |
| question | 6,861 | state questions answered via `get_state`, `music_library`, `get_time` or `timer`: weather now, for the rest of the day and the next days, to-do list items, time and date, timers still running, appliance status sensors, door contacts, open windows and covers of the whole house or a room, electric cars, outdoor temperature, lights of an area, two questions in one sentence, playlists, albums (also of one artist), artists, stations and podcasts of the media library |
| conversation | 4,363 | 3 to 6 turns: undo, corrections, "Welches Licht hast du ausgeschaltet?", recaps, checks, topic changes, music follow-ups ("Lauter", "Zurück", "Vorspulen", pause and play again), playing a playlist named in a library answer, timer follow-ups ("Wie lange noch?"), list follow-ups ("Und Eier"), choosing a speaker after a question back |
| timer | 2,457 | start (also named and with a delayed command such as "Mach in 10 Minuten das Licht aus"), cancel, cancel all, pause, resume, add, subtract, ambiguous names, no matching timer, devices without timer support |
| unknown | 1,533 | devices, rooms or lists that do not exist, alarms at a clock time: no tool call, other rooms suggested, also when the user insists twice; houses without areas |
| error | 1,105 | unreachable devices, partial success, rooms where only some lights are reachable, retries, music not found, players without a media library: the answer reports the failure |
| compound | 473 | two commands in one sentence |
| broadcast | 472 | announcements to all voice devices, also without other devices |
| chat | 411 | greetings, thanks, capabilities, "Vergiss es" |
| followup | 350 | two-turn conversations ("und wieder aus", "Nee, ist es nicht", accepting a suggested other room) |
| relative | 237 | "Mach es im Bad wärmer": read state, then set |
| clarify | 141 | several speakers and none named: the assistant reads the players first, plays on the only reachable one, says that none is reachable, or asks which one |

The generator draws from 117 cases with equal quotas (236 samples each, a few
cases with few distinct sentences have fewer), so rare situations are as
frequent as common ones.

## Music

The system prompt can name the speaker of the voice device ("Der Lautsprecher
des Sprachgeräts: media_player.…"). "Spiel Musik" and other music requests
without a named player play there first, then on the only speaker in the
room of the voice device; otherwise the assistant reads all players and asks.
`music_library` returns the names in the media library of a player
(`media_player.browse_media` in the integration), which
`media_player.search_and_play` then plays by name. `search_and_play` without
a query continues the loaded title or plays the favorite songs, and
`media_player.seek_relative` seeks by seconds (negative: back).

The playlists of a media library are generated per sample like real ones
(`gen/playlists.py`): built-in mixes, artist radios, "This Is ..." lists,
genre and festival playlists, soundtracks and personal names with emojis,
from 2 to 100 entries. For "Spiel meine Lieblingssongs" or "Spiel eine
Playlist zum Einschlafen" the assistant reads the playlists and picks the one
that fits among them; names of related moods are left out, so exactly one
fits. In 12 % of these samples none fits and the assistant names a few others.

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

Created with `gen_dataset.py` and one generator package per language,
`gen/<lang>/` (template-based, no LLM involved):
`python gen_all.py --seed 18 --per-case 236`, which runs
`gen_dataset.py --lang <lang> data/<lang>/train.jsonl per_case:236 18` for
every language and merges the rows. All generators have the same cases, tools
and services (`check_languages.py`); `llm_schema.py` holds the prompt texts
of all languages.

The vendor id prefixes in the generator (Homematic, Shelly) are fictional.
