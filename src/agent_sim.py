"""Replay multi-turn dialogues through the same tool loop as the integration.

The house is simulated: every exposed entity has a state, chosen entities can
be marked unavailable, and control calls change the simulated states. This
shows what the user would hear and what really happened, turn by turn.

Usage: python src/agent_sim.py URL scenarios.json
scenarios.json: {"house": "layout.json" ([[entity_id, name, area], ...]),
                 "scenarios": [{"name": ..., "unavailable": [...],
                                "states": {eid: state}, "turns": [...]}]}
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

MAX_TOOL_ROUNDS = 4
DEFAULT_STATE = {"light": "off", "switch": "off", "cover": "open", "climate": "heat",
                 "media_player": "off", "vacuum": "docked", "lawn_mower": "docked",
                 "valve": "closed", "input_boolean": "off", "binary_sensor": "off",
                 "sensor": "12", "alarm_control_panel": "disarmed", "button": "unknown"}
AFTER = {"turn_on": "on", "turn_off": "off", "open_cover": "open",
         "close_cover": "closed", "start": "cleaning", "start_mowing": "mowing",
         "dock": "docked", "open_valve": "open", "close_valve": "closed",
         "return_to_base": "returning", "lock": "locked", "unlock": "unlocked"}


class House:
    def __init__(self, layout, unavailable=(), states=None):
        self.names = {e: n for e, n, _ in layout}
        self.state = {e: DEFAULT_STATE.get(e.split(".")[0], "unknown") for e in self.names}
        self.state.update(states or {})
        for e in unavailable:
            self.state[e] = "unavailable"
        self.log: list[str] = []

    def get_state(self, args):
        ids = args.get("entity_ids") or []
        out = []
        for eid in ids if isinstance(ids, list) else [ids]:
            if eid not in self.names:
                out.append({"entity_id": eid, "error": "unbekannte entity_id"})
            else:
                out.append(S.format_state(eid, self.names[eid], self.state[eid], {}))
        return S.dumps(out)

    def control(self, args):
        errors, done = [], []
        acts = args.get("actions") or []
        for act in acts if isinstance(acts, list) else [acts]:
            service = act.get("service", "")
            ids = act.get("entity_id") or []
            ids = [ids] if isinstance(ids, str) else ids
            if service not in S.SERVICES:
                errors.append(f"Unknown service {service}")
                continue
            domain, name = service.split(".", 1)
            bad = [i for i in ids if i not in self.names or i.split(".")[0] != domain]
            if bad or not ids:
                errors.append(f"Unknown entity for {service}: {', '.join(bad) or '-'}")
                continue
            unavailable = [i for i in ids if self.state[i] == "unavailable"]
            if unavailable:
                errors.append(f"Entity {', '.join(unavailable)} is unavailable")
                self.log.append(f"FAILED {service} {unavailable} (unavailable)")
                ids = [i for i in ids if i not in unavailable]
                if not ids:
                    continue
            for i in ids:
                if name in AFTER:
                    self.state[i] = AFTER[name]
            done.extend(ids)
            self.log.append(f"{service} {ids}")
        return S.control_result(errors, done)


def complete(url, messages):
    body = json.dumps({"messages": messages, "tools": S.TOOLS, "temperature": 0,
                       "max_tokens": 400}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]


def run(url, layout, sc):
    house = House(layout, sc.get("unavailable", []), sc.get("states"))
    system = S.build_system_prompt([tuple(e) for e in layout])
    history: list[dict] = []
    print(f"\n### {sc['name']}")
    for text in sc["turns"]:
        turn = [{"role": "user", "content": text}]
        house.log = []
        calls_seen = []
        answer = None
        for _ in range(MAX_TOOL_ROUNDS + 1):
            msg = complete(url, [{"role": "system", "content": system}] + history + turn)
            calls = msg.get("tool_calls") or []
            if not calls:
                answer = (msg.get("content") or "").strip()
                turn.append({"role": "assistant", "content": answer})
                break
            turn.append({"role": "assistant", "content": None, "tool_calls": calls})
            for c in calls:
                fn = c["function"]
                args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                res = house.control(args) if fn["name"] == "control" else house.get_state(args)
                calls_seen.append(f"{fn['name']}({json.dumps(args, ensure_ascii=False)}) -> {res}")
                turn.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": res})
        history += turn
        print(f"  USER: {text}")
        for c in calls_seen:
            print(f"    tool: {c[:260]}")
        print(f"  BOT : {answer}")


def main():
    url = sys.argv[1].rstrip("/")
    cfg = json.load(open(sys.argv[2], encoding="utf-8"))
    for sc in cfg["scenarios"]:
        layout = json.load(open(sc.get("house", cfg["house"]), encoding="utf-8"))
        run(url, layout, sc)


if __name__ == "__main__":
    main()
