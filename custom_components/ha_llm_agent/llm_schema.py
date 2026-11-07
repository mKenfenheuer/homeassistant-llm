"""Prompt, tool schema and state format shared by training and runtime.

This module is the single source of truth. The training pipeline (src/)
imports it, and the Home Assistant integration uses it at runtime. It has no
Home Assistant imports so it can run in both environments.

Any change to SYSTEM_TEMPLATE, TOOLS or format_state() changes what the model
sees and requires a new training run.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

SCHEMA_VERSION = 5

# --- services the model may call, with the data keys it may pass ------------
SERVICES: dict[str, list[str]] = {
    "light.turn_on": ["brightness_pct", "brightness_step_pct", "color_name",
                      "color_temp_kelvin"],
    "light.turn_off": [],
    "switch.turn_on": [],
    "switch.turn_off": [],
    "input_boolean.turn_on": [],
    "input_boolean.turn_off": [],
    "fan.turn_on": ["percentage"],
    "fan.turn_off": [],
    "fan.set_percentage": ["percentage"],
    "cover.open_cover": [],
    "cover.close_cover": [],
    "cover.stop_cover": [],
    "cover.set_cover_position": ["position"],
    "climate.set_temperature": ["temperature"],
    "climate.turn_on": [],
    "climate.turn_off": [],
    "lock.lock": [],
    "lock.unlock": [],
    "alarm_control_panel.alarm_arm_home": ["code"],
    "alarm_control_panel.alarm_arm_away": ["code"],
    "alarm_control_panel.alarm_arm_night": ["code"],
    "alarm_control_panel.alarm_disarm": ["code"],
    "media_player.turn_on": [],
    "media_player.turn_off": [],
    "media_player.media_play": [],
    "media_player.media_pause": [],
    "media_player.media_stop": [],
    "media_player.media_next_track": [],
    "media_player.media_previous_track": [],
    "media_player.volume_set": ["volume_level"],
    "media_player.volume_up": [],
    "media_player.volume_down": [],
    "vacuum.start": [],
    "vacuum.pause": [],
    "vacuum.return_to_base": [],
    "scene.turn_on": [],
    "script.turn_on": [],
    "button.press": [],
    "lawn_mower.start_mowing": [],
    "lawn_mower.pause": [],
    "lawn_mower.dock": [],
    "valve.open_valve": [],
    "valve.close_valve": [],
}

# domains that appear in the device list (controllable + read-only)
CONTROL_DOMAINS = sorted({s.split(".", 1)[0] for s in SERVICES})
READ_DOMAINS = ["sensor", "binary_sensor", "weather"]
EXPOSED_DOMAINS = CONTROL_DOMAINS + READ_DOMAINS
# domains that can be addressed by area ("das Licht in der Küche", "alle
# Lichter"). Covers are left out: "die Rollläden auf der Terrasse" must not
# move the awning in the same area.
AREA_DOMAINS = ["light"]
ALL_AREAS = "*"

SYSTEM_TEMPLATE = (
    "Du bist der Sprachassistent eines Smart Homes mit Home Assistant. "
    "Antworte immer auf Deutsch, kurz und freundlich.\n"
    "Steuere Geräte mit dem Werkzeug control. Frage Zustände und Messwerte "
    "mit get_state ab, bevor du Fragen dazu beantwortest.\n"
    "Verwende nur entity_ids aus der folgenden Liste. Gibt es ein Gerät "
    "nicht, sage das, statt eine entity_id zu erfinden.\n"
    "Meint der Nutzer alle Lichter eines Bereichs, gib statt entity_id den "
    "Bereich an (area), für das ganze Haus area \"*\".\n\n"
    "Geräte (entity_id | Name | Bereich):\n{devices}"
)
LOCATION_TEMPLATE = "\n\nDas Sprachgerät steht im Bereich: {area}"

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "control",
            "description": (
                "Steuert ein oder mehrere Geräte. Jede Aktion ruft einen "
                "Home-Assistant-Dienst für eine oder mehrere entity_ids auf, "
                "bei Lichtern auch für alle Lichter eines Bereichs (area). "
                "Erlaubte Dienste: " + ", ".join(SERVICES) + "."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "service": {
                                    "type": "string",
                                    "description": "Dienst im Format domain.dienst",
                                },
                                "entity_id": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "area": {
                                    "type": "string",
                                    "description": (
                                        "Statt entity_id: alle Lichter dieses "
                                        "Bereichs, \"*\" für das ganze Haus"
                                    ),
                                },
                                "data": {
                                    "type": "object",
                                    "description": (
                                        "Optionale Parameter, z. B. brightness_pct, "
                                        "brightness_step_pct, color_name, "
                                        "color_temp_kelvin, position, temperature, "
                                        "percentage, volume_level, code"
                                    ),
                                },
                            },
                            "required": ["service"],
                        },
                    }
                },
                "required": ["actions"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_state",
            "description": (
                "Liefert den aktuellen Zustand und wichtige Attribute von "
                "Geräten und Sensoren, entweder für entity_ids oder für alle "
                "Lichter eines Bereichs (domain light und area)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_ids": {"type": "array", "items": {"type": "string"}},
                    "domain": {"type": "string"},
                    "area": {"type": "string"},
                },
            },
        },
    },
]

# attributes passed back to the model by get_state
STATE_ATTRS = [
    "device_class", "unit_of_measurement", "current_temperature", "temperature",
    "hvac_action", "current_position", "percentage", "volume_level",
    "media_title", "media_artist",
]


def device_line(entity_id: str, name: str, area: str | None) -> str:
    return f"{entity_id} | {name} | {area or '-'}"


def sort_key(entity: tuple[str, str, str | None]) -> tuple[str, str]:
    """Entities are listed grouped by area, then by entity_id."""
    return (entity[2] or "~", entity[0])


def build_system_prompt(entities: Iterable[tuple[str, str, str | None]],
                        location: str | None = None) -> str:
    """entities: (entity_id, name, area_name) tuples."""
    lines = "\n".join(device_line(*e) for e in sorted(entities, key=sort_key))
    prompt = SYSTEM_TEMPLATE.format(devices=lines)
    if location:
        prompt += LOCATION_TEMPLATE.format(area=location)
    return prompt


def format_state(entity_id: str, name: str, state: str,
                 attrs: dict[str, Any]) -> dict[str, Any]:
    """One entry of a get_state result."""
    out: dict[str, Any] = {"entity_id": entity_id, "name": name, "state": state}
    if attrs.get("brightness") is not None and state == "on":
        out["brightness_pct"] = round(attrs["brightness"] / 255 * 100)
    for key in STATE_ATTRS:
        val = attrs.get(key)
        if val is None or val == "":
            continue
        if isinstance(val, float):
            val = round(val, 2)
        out[key] = val
    return out


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def control_result(errors: list[str] | None = None,
                   done: list[str] | None = None) -> str:
    """Result of a control call. `done` lists the entity_ids that were
    switched: on failure the successful part, on success the entities an
    area resolved to (callers pass it on success only for area actions)."""
    if errors:
        out: dict[str, Any] = {"success": False, "error": "; ".join(errors)}
        if done:
            out["done"] = done
        return dumps(out)
    if done:
        return dumps({"success": True, "done": done})
    return dumps({"success": True})


def area_targets(domain: str, area: str,
                 entities: Iterable[tuple[str, str, str | None]],
                 groups: Iterable[str] = ()) -> list[str] | None:
    """entity_ids of `domain` in `area` ("*" = whole house), in device-list
    order. Group entities are left out, their members are listed anyway.
    None if the area does not exist."""
    ents = sorted(entities, key=sort_key)
    groups = set(groups)
    if area != ALL_AREAS:
        names = {a for _, _, a in ents if a}
        match = next((a for a in names if a == area), None) or \
            next((a for a in names if a.casefold() == area.strip().casefold()), None)
        if match is None:
            return None
        area = match
    return [e for e, _, a in ents
            if e.split(".", 1)[0] == domain and e not in groups
            and (area == ALL_AREAS or a == area)]
