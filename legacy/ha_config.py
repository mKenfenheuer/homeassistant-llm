"""Schema + fake house config shared by the dataset generators.

The fake house mirrors a realistic German Home Assistant setup so the model
learns name->entity_id mapping, tool calling, and grounded state questions.
At inference time (integration/ha_agent.py) the same tool schema is built
from the REAL HA entity list, so the skills transfer.
"""

import json

# ---- fake house --------------------------------------------------------------
# entity_id, friendly name (German), domain, state fields
ENTITIES = [
    ("light.esszimmer",           "Esszimmer Himmelslicht",  "light",
     {"state": "on", "brightness_pct": 60}),
    ("light.esszimmer_stehlampe", "Stehlampe Esszimmer",     "light",
     {"state": "off"}),
    ("light.livingroom_ceiling",  "Livingroom Himmelslicht", "light",
     {"state": "off"}),
    ("light.kitchen",             "Kitchen Lights",          "light",
     {"state": "on", "brightness_pct": 100}),
    ("light.bedroom",             "Bedroom Lights",          "light",
     {"state": "off"}),
    ("light.bathroom",            "Bathroom Lights",         "light",
     {"state": "on", "brightness_pct": 80}),
    ("light.garden_floodlights",  "Garten Floodlights",      "light",
     {"state": "off"}),
    ("light.flur",                "Hall Lights",             "light",
     {"state": "off"}),

    ("switch.washing_machine",    "Washing Machine",         "switch",
     {"state": "off"}),
    ("switch.garten_teichpump",   "Teichpump",        "switch",
     {"state": "on"}),

    ("cover.livingroom_laden",    "Livingroom Shutters",     "cover",
     {"state": "open"}),
    ("cover.esszimmer_laden",     "Esszimmer Shutters",      "cover",
     {"state": "closed"}),
    ("cover.bedroom_laden",       "Bedroom Shutters",        "cover",
     {"state": "closed"}),
    ("cover.bathroom_laden",      "Bathroom Shutters",       "cover",
     {"state": "open"}),

    ("alarm_control_panel.alarm", "Alarm System",            "alarm_control_panel",
     {"state": "disarmed"}),

    ("thermostat.bathroom_heating", "Bathroom Radiator",     "thermostat",
     {"temperature": 22.0, "target_temperature": 21.0}),
    ("thermostat.livingroom_heating", "Livingroom Radiator", "thermostat",
     {"temperature": 21.5, "target_temperature": 20.0}),

    ("sensor.bedroom_temperature",   "Bedroom Thermometer",  "sensor",
     {"value": 21.5, "unit": "°C"}),
    ("sensor.livingroom_temperature", "Livingroom Thermometer", "sensor",
     {"value": 22.0, "unit": "°C"}),
    ("sensor.aussen_temperature",    "Aussen Thermometer",   "sensor",
     {"value": 11.0, "unit": "°C"}),
    ("sensor.livingroom_humidity",   "Livingroom Hygrometer", "sensor",
     {"value": 43.0, "unit": "%"}),
]

ENTITY_INFO = {e[0]: {"name": e[1], "domain": e[2], "state": e[3]} for e in ENTITIES}
BY_DOMAIN = {}
for eid, info in ENTITY_INFO.items():
    BY_DOMAIN.setdefault(info["domain"], []).append(eid)

# actions each domain supports, with required data params
ACTIONS = {
    "light": {
        "light.turn_on": {},
        "light.turn_off": {},
        "light.set_brightness": {"brightness_pct": (0, 100)},
        "light.set_color": {"color_hex": "#rrggbb"},
    },
    "switch": {"switch.turn_on": {}, "switch.turn_off": {}},
    "cover": {"cover.open": {}, "cover.close": {}, "cover.stop": {}},
    "alarm_control_panel": {
        "alarm_panel.arm_home": {},
        "alarm_panel.arm_away": {},
        "alarm_panel.arm_night": {},
        "alarm_panel.arm_vacation": {},
        "alarm_panel.disarm": {},
    },
    "thermostat": {"thermostat.set_temperature": {"temperature": (5, 35)}},
    "sensor": {},
}

ALL_ACTION_NAMES = [a for d in ACTIONS.values() for a in d]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_states",
            "description": "Returns the current state of all entities in the home. Use this to answer questions about the home.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_attributes",
            "description": "Returns the attributes and current state of one entity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "enum": list(ENTITY_INFO)},
                },
                "required": ["entity_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "control",
            "description": "Control entities: lights, switches, shutters (covers), the alarm panel and thermostats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ALL_ACTION_NAMES},
                                "entity_id": {"type": "string", "enum": list(ENTITY_INFO)},
                                "data": {"type": "object", "description": "action parameters, e.g. brightness_pct, color_hex, temperature"},
                            },
                            "required": ["action", "entity_id"],
                        },
                    }
                },
                "required": ["actions"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a home assistant. Answer in German. You control the smart home with "
    "functions and answer questions about the state of the home. If the user mentions "
    "an entity that does not exist, say so and never make up an entity_id."
)


def state_summary():
    """The dict a real HA `get_states` call returns (friendly name -> state)."""
    out = {}
    for eid, info in ENTITY_INFO.items():
        st = dict(info["state"])
        st["_name"] = info["name"]
        out[eid] = st
    return out


def tool_result(name, args):
    """Fake tool execution result for the training data."""
    if name == "get_states":
        return json.dumps(state_summary(), ensure_ascii=False)
    if name == "get_entity_attributes":
        return json.dumps(
            {args["entity_id"]: {**ENTITY_INFO[args["entity_id"]]["state"],
                                 "_name": ENTITY_INFO[args["entity_id"]]["name"]}},
            ensure_ascii=False)
    if name == "control":
        return json.dumps({"ok": True})
    raise ValueError(name)


