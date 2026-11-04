"""Generate the template-based part of the German SFT dataset (JSONL).

Every row is one conversation in OpenAI chat format:
  {"messages": [ {role, content, [tool_calls]} ... ]}
Assistant tool calls use the `control` / `get_states` tools from ha_config.
Gold answers are computed from the fake house state, so they are exact.
"""

import json
import random

import ha_config
from ha_config import ENTITY_INFO, BY_DOMAIN, SYSTEM_PROMPT

STATE = {eid: info["state"] for eid, info in ENTITY_INFO.items()}
NAME = {eid: info["name"] for eid, info in ENTITY_INFO.items()}
DOMAIN = {eid: info["domain"] for eid, info in ENTITY_INFO.items()}

COLORS = {"red": "#ff0000", "green": "#00ff00", "blue": "#0000ff",
          "orange": "#ff9900", "pink": "#ff66aa", "white": "#ffffff"}

ON = ["schlicht {n}", "schlicht {n}", "schlicht {n}", "schlicht {n}",
      "schlicht {n}, please", "schlicht {n}", "please schlicht {n}"]
OFF = ["schlicht off {n}", "schlicht off {n}", "schlicht off {n}",
       "turn off {n}", "turn off {n}"]
COVER_OPEN = ["open {n}", "open {n}", "open {n}", "raise {n}",
              "open again {n}", "open again the {n}"]
COVER_CLOSE = ["close {n}", "close {n}", "close {n}", "close {n}",
               "close {n}", "close {n}", "lower {n}"]
COVER_STOP = ["stop the {n}", "stop the {n}", "stop the {n}", "stop {n}"]
CONFIRM = ["Okay, {verb_past} {n}.", "Okay, {n} {verb_past}.",
           "Okay, done.", "Okay, {verb_past} {n}!", "Done, {verb_past} {n}."]
PAST = {"light.turn_on": "switched on", "light.turn_off": "turned off",
        "switch.turn_on": "switched on", "switch.turn_off": "turned off",
        "cover.open": "opened", "cover.close": "closed", "cover.stop": "stopped",
        "alarm_panel.arm_home": "armed the alarm with home mode",
        "alarm_panel.arm_away": "armed the alarm with away mode",
        "alarm_panel.arm_night": "armed the alarm with night mode",
        "alarm_panel.arm_vacation": "armed the alarm with vacation mode",
        "alarm_panel.disarm": "disarmed the alarm",
        "light.set_brightness": "set", "light.set_color": "colored",
        "thermostat.set_temperature": "set"}

UNKNOWN = ["schlicht {u}", "open {u}", "turn off {u}", "close {u}",
           "can you switch on {u}", "activate {u}"]
UNKNOWN_NAMES = ["dishwasher", "coffee machine", "terrace shutters", "garage",
                 "sauna", "pool pump", "air conditioner", "fan"]
UNKNOWN_ANSWERS = [
    "Sorry, there is no {u} in this house.",
    "I can't do that: no {u} is connected.",
    "There is no {u} that I know of.",
]


def make_control_seed(action, eid, data=None):
    """One single-action seed: user says X -> gold tool call + confirmation."""
    if action == "light.set_brightness":
        user = random.choice([
            "set {n} to {d} percent", "turn {n} to {d} percent",
            "dim {n} to {d} percent", "set {n} to {d} %",
        ]).format(n=NAME[eid], d=data["brightness_pct"])
    elif action == "light.set_color":
        cname = [k for k, v in COLORS.items() if v == data["color_hex"]][0]
        user = random.choice([
            "set {n} to {c}", "please set {n} to {c}",
        ]).format(n=NAME[eid], c=cname)
    elif action.startswith("light.") or action.startswith("switch."):
        user = random.choice(ON if action.endswith("turn_on") else OFF)
    elif action == "cover.open":
        user = random.choice(COVER_OPEN)
    elif action == "cover.close":
        user = random.choice(COVER_CLOSE)
    elif action == "cover.stop":
        user = random.choice(COVER_STOP)
    elif action == "alarm_panel.disarm":
        user = random.choice(["disarm the alarm", "deactivate the alarm!",
                              "I'm back, disarm the alarm",
                              "can you deactivate the alarm?"])
    elif action.startswith("alarm_panel.arm"):
        mode = action.split(".")[1].replace("arm_", "")
        user = random.choice([
            "activate the alarm!", "alarm!", "activate the alarm!",
            "activate the alarm with {mode} mode".format(mode=mode),
            "let's go: activate the alarm with {mode} mode".format(mode=mode),
        ])
    elif action == "light.set_brightness":
        user = random.choice([
            "set {n} to {d} percent", "turn {n} to {d} percent",
            "dim {n} to {d} percent", "set {n} to {d} %",
        ]).format(n=NAME[eid], d=data["brightness_pct"])
    elif action == "light.set_color":
        c = [k for k, v in COLORS.items() if v == data["color_hex"]][0]
        user = random.choice([
            "set {n} to {c}", "please set {n} to {c}",
        ]).format(n=NAME[eid], c=c)
    elif action == "thermostat.set_temperature":
        user = random.choice([
            "set {n} to {d} degrees", "set {n} to {d} degrees",
            "turn {n} up to {d} degrees",
        ]).format(n=NAME[eid], d=data["temperature"])
    else:
        raise ValueError(action)
    user = user.format(n=NAME[eid])
    return {"user": user,
            "actions": [{"action": action, "entity_id": eid,
                         "data": data or {}}]}


def make_multi_seed(eids, action, group_noun, user_tpls):
    """One multi-action seed: a group command, e.g. 'close all shutters'."""
    act = {"action": action, "data": {}}
    return {"user": random.choice(user_tpls),
            "actions": [{**act, "entity_id": e} for e in eids]}


def render(seed):
    """Seed (user text + actions) -> full gold conversation messages."""
    call = {"role": "assistant",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "control",
                                         "arguments": json.dumps(
                                             {"actions": seed["actions"]},
                                             ensure_ascii=False)}}]}
    acts = seed["actions"]
    if len(acts) == 1:
        a = acts[0]
        conf = random.choice(CONFIRM).format(
            verb_past=PAST[a["action"]], n=NAME[a["entity_id"]])
        if a["action"] == "light.set_brightness":
            conf = random.choice([
                "Okay, {n} at {d} percent.",
                "Okay, {n} to {d} percent.",
            ]).format(n=NAME[a["entity_id"]], d=a["data"]["brightness_pct"])
        if a["action"] == "light.set_color":
            cname = [k for k, v in COLORS.items() if v == a["data"]["color_hex"]][0]
            conf = "Okay, {n} set to {c}.".format(n=NAME[a["entity_id"]], c=cname)
        if a["action"] == "thermostat.set_temperature":
            conf = random.choice([
                "Okay, {n} set to {d} degrees.",
                "Okay, {n} to {d} degrees.",
            ]).format(n=NAME[a["entity_id"]], d=a["data"]["temperature"])
        if a["action"].startswith("alarm_panel."):
            conf = random.choice([
                "Okay, the alarm is activated.", "Okay, the alarm is activated!",
                "Okay, done!", "Okay, the alarm is deactivated.",
            ]) if a["action"] == "alarm_panel.disarm" else random.choice([
                "Okay, the alarm is activated!", "Okay, the alarm is activated.",
                "Okay, done!",
            ])
    else:
        conf = random.choice([
            "Okay, done!", "Okay, all done!", "Done, done!",
        ])
    result = ha_config.tool_result("control", {"actions": acts})
    return [{"role": "user", "content": seed["user"]}, call,
            {"role": "tool", "tool_call_id": "call_1", "content": result},
            {"role": "assistant", "content": conf}]


def make_qa_seeds():
    """State questions. Two forms:
    (a) state pre-loaded in the system prompt -> answer directly;
    (b) state only via get_states tool call."""
    rows = []
    sensors = [e for e in ENTITY_INFO if DOMAIN[e] == "sensor"]
    temps = [e for e in sensors if "temp" in e]
    hums = [e for e in sensors if "humidity" in e]

    def room_of(e):
        return NAME[e].replace("Thermometer", "").replace("Hygrometer", "").strip()

    qa = []
    for e in temps:
        v = ("%g" % STATE[e]["value"]).replace(".", ",")
        if "aussen" in e:
            qa.append(("What temperature is it outside?",
                       "It's {v} degrees outside.".format(v=v)))
            qa.append(("How warm is it outside?",
                       "It's currently {v} degrees outside.".format(v=v)))
            continue
        room = room_of(e)
        qa.append(("What temperature is it in {room}?".format(room=room),
                   "It's {v} degrees in {room}.".format(v=v, room=room)))
        qa.append(("How warm is it in {room}?".format(room=room),
                   "{room} is {v} degrees warm.".format(room=room, v=v)))
    for e in hums:
        v = ("%g" % STATE[e]["value"]).replace(".", ",")
        qa.append(("How high is the humidity in {room}?".format(room=room_of(e)),
                   "The humidity in {room} is {v} percent.".format(room=room_of(e), v=v)))

    on_lights = [NAME[e] for e in ENTITY_INFO
                 if DOMAIN[e] == "light" and STATE[e].get("state") == "on"]
    qa.append(("Which lights are on?",
               "Currently on: {l}.".format(l=", ".join(on_lights))))
    qa.append(("Which lights are still lit?",
               "The following lights are on: {l}.".format(l=", ".join(on_lights))))
    qa.append(("Is the alarm armed?",
               "No, the alarm system is currently disarmed."))
    open_covers = [NAME[e] for e in ENTITY_INFO
                   if DOMAIN[e] == "cover" and STATE[e].get("state") == "open"]
    if open_covers:
        qa.append(("Are all shutters closed?",
                   "No, the following are open: {l}.".format(l=", ".join(open_covers))))
    else:
        qa.append(("Are all shutters closed?", "Yes, all shutters are closed."))
    return rows, qa


def build_rows():
    rng = random.Random(13)
    rows = []

    # 1) single control actions over controllable domains
    for eid, info in ENTITY_INFO.items():
        d = DOMAIN[eid]
        for action in ha_config.ACTIONS[d]:
            data = None
            if action == "light.set_brightness":
                data = {"brightness_pct": rng.choice([10, 25, 40, 60, 80, 100])}
            elif action == "light.set_color":
                name, hexv = rng.choice(list(COLORS.items()))
                data = {"color_hex": hexv}
            elif action == "thermostat.set_temperature":
                data = {"temperature": rng.choice([18, 19, 20, 21, 22, 23])}
            seed = make_control_seed(action, eid, data)
            rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + render(seed)})

    # 2) group commands
    groups = [
        (BY_DOMAIN["light"], "light.turn_on",
         ["turn on all lights!", "it's dark, turn on all lights",
          "turn on all lights", "please turn on all lights"]),
        (BY_DOMAIN["light"], "light.turn_off",
         ["turn off all lights!", "turn off all lights",
          "we're leaving, turn off all lights", "turn off all the lights"]),
        (BY_DOMAIN["cover"], "cover.close",
         ["close all shutters!", "close all shutters",
          "it's sunny, close all shutters", "please close all shutters"]),
        (BY_DOMAIN["cover"], "cover.open",
         ["open all shutters", "open all shutters, please",
          "open the shutters again"]),
    ]
    for eids, action, tpls in groups:
        seed = make_multi_seed(eids, action, None, tpls)
        rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + render(seed)})

    # 3) unknown entity -> refuse (never invent an entity_id)
    for u in rng.sample(UNKNOWN_NAMES, 5):
        user = random.choice(UNKNOWN).format(u=u)
        ans = random.choice(UNKNOWN_ANSWERS).format(u=u)
        rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": user},
                                  {"role": "assistant", "content": ans}]})

    # 4) follow-up anaphora: "... now turn it off"
    for _ in range(40):
        eid = rng.choice([e for e in ENTITY_INFO if DOMAIN[e] in ("light", "cover")])
        act_on = ("light.turn_on" if DOMAIN[eid] == "light" else "cover.open")
        act_off = ("light.turn_off" if DOMAIN[eid] == "light" else "cover.close")
        msgs = render(make_control_seed(act_on, eid))
        follow = random.choice(
            ["now turn it off", "turn it off", "turn it off",
             "turn it off now"] if DOMAIN[eid] == "light" else
            ["close it", "close it again", "close it now", "close it again"])
        second = render(make_control_seed(act_off, eid))
        msgs += [{"role": "user", "content": follow}] + second[1:]
        rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + msgs})

    # 5) state Q&A, with and without the state preloaded
    _, qa_pairs = make_qa_seeds()
    state_json = json.dumps(ha_config.state_summary(), ensure_ascii=False)
    for user_q, gold in qa_pairs:
        # (b) tool-based answering
        call = {"role": "assistant",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "get_states",
                                             "arguments": "{}"}}]}
        rows.append({"messages": [{"role": "system", "content": SYSTEM_PROMPT},
                                  {"role": "user", "content": user_q},
                                  call,
                                  {"role": "tool", "tool_call_id": "call_1",
                                   "content": state_json},
                                  {"role": "assistant", "content": gold}]})
        # (a) state preloaded in system prompt -> direct answer
        rows.append({"messages": [{"role": "system",
                                   "content": SYSTEM_PROMPT +
                                   "\n\nCurrent home state (JSON): " + state_json},
                                  {"role": "user", "content": user_q},
                                  {"role": "assistant", "content": gold}]})

    random.Random(7).shuffle(rows)
    return rows


def main():
    rows = build_rows()
    out = "../data/template.jsonl"  # placeholder, overridden below
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/template.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "rows to", path)


if __name__ == "__main__":
    main()
