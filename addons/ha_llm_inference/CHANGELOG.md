# Changelog

## 0.14.0

- Model v17 (prompt schema 7), trained on from v16. Results: 97.7 % correct
  tool calls on 300 held-out conversations (v16: 94.7 %), 53 of 55 turns in
  the test on a real house (v16: 45 of 55).
- Media library: "Welche Playlists habe ich?", "Welche Alben von Queen habe
  ich?", "Spiel eine Playlist zum Einschlafen". The integration reads the
  library of the player with `media_player.browse_media` (new tool
  `music_library`), without Music Assistant.
- "Spiel Musik" goes on with the loaded title, otherwise it plays the
  favorite songs. "Spiel meine Lieblingssongs" switches to the favorites
  playlist.
- The speaker of the voice device comes first: "Spiel Musik" and "Pause"
  without a player play there. Without one the assistant reads all players,
  plays on the only reachable one or asks.
- New: play, seeking forward and back ("Vorspulen", "Spul 30 Sekunden
  zurück"), "Weiter" and "Zurück" as next and previous track.
- Short media commands and started music are not confirmed, at any answer
  length; only a failure is spoken.
- Music that the wake word stopped goes on after the answer (option "Continue
  music after the answer", on by default).
- "Welche Rollläden sind offen?" and "Sind alle Fenster zu?" read all covers
  or windows of the house or a room; two questions in one sentence get both
  answers.
- Speech-to-text sends a short command vocabulary to whisper, so "Nächster
  Titel" is no longer "Nexter Tuttle".
- Update the integration to 0.11.0 together with this app: model v17 needs
  its prompt and tools. Turn on "Use the area and the speaker of the voice
  device" in the integration options so that music plays on the voice
  device.

## 0.13.0

- Model v16 (prompt schema 6). New: music (Music Assistant or the media
  search of the player), timers on voice devices (start, pause, resume,
  add, subtract, cancel, status, delayed commands), time and date,
  broadcasts to all speakers, shopping and to-do lists, humidifiers and
  valve positions. Weather answers name the condition and the rest of the
  day. Answer length set in the integration (short, medium, long,
  detailed).
- All languages share the Hugging Face repository
  `mkenfenheuer/qwen2.5-3b-homeassistant-GGUF`; `models.json` links each
  language to its model file. German is available today.
- Update the integration to 0.10.0 together with this app: model v16 needs
  its prompt and tools.

## 0.12.0

- New option `gpu_memory`: a budget for the GPU memory in MiB. The app keeps
  at most this much and puts the remaining layers on the CPU.
- If another app holds the GPU memory, the app no longer stops with "unable to
  allocate CUDA0 buffer". It waits, tries again three times, then starts with
  fewer layers on the GPU and at last on the CPU.
- HA TTS 0.2.0 with a fixed context of 4096 tokens: started first, it took
  128k tokens and so much GPU memory that the language model didn't fit.

## 0.11.0

- New option `language`: the app downloads the model of the selected
  language from `mkenfenheuer/qwen2.5-3b-homeassistant-<language>-GGUF`.
  Available today: German (`de`). Models of other languages go to
  `/share/ha_llm/<language>/`.
- The CUDA cache is kept in `/data` across restarts. GPUs without prebuilt
  code (e.g. Pascal) don't compile it again on every start.
- Documentation, option descriptions and log messages in English. The
  settings are translated into English and German.
- New app "HA Whisper" in this repository: speech-to-text with whisper.cpp on
  the GPU or the CPU.
- Integration 0.8.0, now "HA LLM Agent (local)": adding the integration asks
  for the kind of entry, conversation agent or speech-to-text. The
  speech-to-text entry uses the HA Whisper app and sends the language of the
  Assist pipeline. Existing entries stay conversation agents.

## 0.10.0

- One app for CPU and GPU: "HA LLM Inference" uses an NVIDIA GPU (CUDA), an
  AMD or Intel GPU (Vulkan) or the CPU, depending on what it finds. The apps
  "HA LLM Inference (NVIDIA)" and "(AMD/Intel)" are discontinued. If you use
  one of them, install "HA LLM Inference" and uninstall the old one; the
  integration finds the app during setup, and you can switch an existing
  entry over with "Reconfigure". The model under `/share/ha_llm/` stays.
- New option `gpu_layers` (default 99 = all layers, 0 = CPU only).
- The image is larger on amd64 (about 5.9 GB instead of 1.2 GB) because it
  contains the CUDA runtime.

## 0.9.1

- The AVX2 warning now appears only on x86 CPUs, no longer on ARM.
- The GPU apps run on the CPU when no GPU is present and get the GPU through
  the `video` option.

## 0.9.0

- Two new apps for GPUs in the Home Assistant host, both experimental:
  "HA LLM Inference (NVIDIA)" with CUDA 12.8 (Maxwell and newer, tested with a
  GTX 1060) and "HA LLM Inference (AMD/Intel)" with Vulkan (tested with a
  Radeon RX 6900 XT, 130 tokens per second). They need Home Assistant OS
  with GPU support for apps (option `gpu`), which so far exists only as a
  test image. New option `gpu_layers` (default 99 = all layers).
- Integration 0.7.2: During setup it picks the GPU app if one is installed,
  otherwise the CPU app.
- CPU app unchanged except for the shared start script.

## 0.8.1

- Custom icon and logo for app and integration. The integration ships them
  under `custom_components/ha_llm_agent/brand/`, and Home Assistant 2026.3
  and newer shows them without an entry in the brands repository
  (integration 0.7.1). The SVG sources are under `brand/`.
- The model and dataset cards on Hugging Face link to this repository with
  the app and integration.

## 0.8.0

- Automatic model updates: With `auto_update` (default: on), the app checks
  the file `latest.json` on Hugging Face at startup and every 6 hours
  (`update_interval`). It downloads a new model while running, verifies the
  SHA-256 checksum, switches to it and deletes the old one. Without a network
  it starts with the last downloaded model. A custom `model_url` or
  `model_path` turns updates off.
- Model v15.1 (`qwen2.5-3b-ha-v15.1-Q4_K_M.gguf`): Same weights as v15-soup;
  without a system message, the embedded chat template uses a German default
  system prompt instead of "You are Qwen". With a system message, as the
  integration sends it, the prompt does not change. Test results are
  identical: 299 of 300 tool calls, 24 of 24 in the Home Assistant test.
- Hugging Face now hosts only the current model.

## 0.7.1

- New model v15-soup (`qwen2.5-3b-ha-v15-soup-Q4_K_M.gguf`). It averages
  the LoRA weights of the three training runs v13, v14 and v15 ("Model Soup").
  Each single run made its own individual errors; averaging evens them out.
  On 300 test conversations, 99.7 % of tool calls are correct (v11:
  98.0 %).
- Quantization with an importance matrix from our own training conversations.
  Without it, Q4_K_M damaged individual runs (v13: 291 of 300 unquantized,
  243 of 300 in Q4). The file stays 1.9 GB and equally fast.
- "Mach den Fernseher im Schlafzimmer an" now names the bedroom instead of
  the media player of a voice assistant in the room.
- "Bewässere die Balkonpflanzen" opens the valve on the balcony, even when a
  button "Bewässerung Garten" without an area is listed next to it.
- "Welche Heizungen hast du verstellt?" names all heaters changed in the
  conversation, not only the last one.
- A question about a room with partly unavailable lamps names all
  unavailable lamps, including the one named after the room.
- Integration unchanged; the model still uses schema 5 (integration
  0.7.0).

## 0.7.0

- New model v11 (`qwen2.5-3b-ha-v11-Q4_K_M.gguf`), retrained from
  Qwen2.5-3B-Instruct with 8,000 examples and 2.5 epochs. Each of the 64
  cases in the generator has the same share, rare cases included.
- Schema 5: "Mach das Licht in der Küche an" switches all lights in the
  area through `area`, "alle Lichter" through `area: "*"`. The integration
  resolves the area itself and reports the switched lamps back. Before, the
  model had to copy every entity_id and with four lamps occasionally left
  one out.
- `get_state` queries all lights in an area with `domain` and `area`.
- Door contacts and electric cars are in the training data: "Ist die
  Balkontür offen?" queries the balcony door contact, not the front door,
  and "Wie weit kommt das Auto noch?" the range instead of the charge level.
- Outdoor temperature next to forecast, dew point and feels-like temperature
  sensors: "Wie warm ist es draußen?" reads the outdoor sensor, "Wie warm
  wird es morgen?" the forecast.
- The correction "Nein, ich meinte im Esszimmer" turns off all kitchen
  lamps, not just one.
- App: An update to the new default model deletes the model files of
  earlier versions in `/share/ha_llm/` (before, 1.9 GB per version stayed
  behind).
- Integration and model belong together: integration 0.7.0 needs model
  v11; older models do not know `area`.

## 0.6.2

- Integration: "Reconfigure" reloads the entry through its update listener.
  HA used to warn because the flow also reloaded it itself; from HA 2026.12
  on this would be an error.

## 0.6.1

- Integration: You can change the server URL with "Reconfigure" on the
  entry, for example when moving from the app to llama-server on a GPU
  machine. The integration checks the new URL through `/health` before it
  applies it. Agent and options are kept. Before, the entry had to be deleted
  and created again.
- App and model (v7) are unchanged.

## 0.6.0

- New model v7 (`qwen2.5-3b-ha-v7-Q4_K_M.gguf`), retrained from
  Qwen2.5-3B-Instruct with 8,000 examples and 2.5 epochs.
- Rooms with partly unavailable lamps: "Ist das Licht in der Küche an?"
  names the available lamps with their state and the rest as unavailable.
  Before, the model answered "Alle Lichter sind aus".
- "Mach das Licht in der Küche an" with partly unavailable lamps switches
  the others and names both parts.
- If the user insists several times on a device that is not in the list,
  the model refuses each time instead of inventing an entity_id.
- Heating mode switches: "Aktiviere den Abwesenheitsmodus der Heizung"
  switches the switch "Heizung … Abwesenheitsmodus", not the thermostat.
- "das Licht im Flur" always means all lamps in the room. Only the name
  alone ("Mach Küche an") means the lamp named like the room.
- Known weakness: Questions about door contacts ("Ist die Balkontür im
  Wohnzimmer offen?") sometimes hit the wrong contact, depending on the
  device list.

## 0.5.0

- New model v4 (`qwen2.5-3b-ha-v4-Q4_K_M.gguf`), retrained from
  Qwen2.5-3B-Instruct with 8,000 examples and 2.5 epochs.
- "Licht in der Küche" switches all lamps in the room, even when one lamp
  itself is called "Küche" and "Küche Links" and "Küche Rechts" are next to
  it.
- Multiple irrigation systems: "Bewässere die Balkonpflanzen" opens the valve
  on the balcony and does not start the garden irrigation.
- Partial successes name both parts: "Gartenbeleuchtung ist gerade nicht
  erreichbar. Den Rest habe ich erledigt: Flur eingeschaltet."
- Check questions like "Ist es jetzt an?" start with "Ja" or "Nein",
  matching the state.
- Knows device states like `in_use`, `program_ended` or `not_connected`.
- Integration: If an action contains several devices and one of them is
  unavailable, it switches the others and reports only that one as an error.
  Before, the whole action failed.

## 0.4.0

- New model v3 (`qwen2.5-3b-ha-v3-Q4_K_M.gguf`), retrained from
  Qwen2.5-3B-Instruct.
- Conversations over several turns: "Und wieder aus", "Welches Licht hast du
  ausgeschaltet?", "Nein, ich meinte im Esszimmer", "Mach das rückgängig".
- If a device is unavailable, the model reports the error instead of
  claiming success. With several commands it reports that only some worked.
- The model refuses devices that are not in the list instead of switching a
  similar device. If the device exists in another room, it names that room.
- Tells apart the status sensor and the smart plug of a device ("Trockner"
  and "Trockner Power") and knows awnings, robot mowers, valves, groups and
  buttons.
- Empty options `model_url` and `model_sha256` load the model that belongs
  to the app version. An app update thus also brings the new model.
  Older default models are removed after the download.

## 0.3.1

- Warning in the log and in Home Assistant if the CPU does not support AVX2.

## 0.3.0

- First version with download from Hugging Face and SHA-256 verification.
