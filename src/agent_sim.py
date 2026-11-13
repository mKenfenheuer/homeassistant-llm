"""Replay multi-turn dialogues through the same tool loop as the integration.

The house is simulated: every exposed entity has a state, chosen entities can
be marked unavailable, and control calls change the simulated states. This
shows what the user would hear and what really happened, turn by turn.

Usage: python src/agent_sim.py URL scenarios.json
scenarios.json: {"house": "layout.json" ([[entity_id, name, area], ...]),
                 "groups": [group entity_ids, left out of area targets],
                 "scenarios": [{"name": ..., "unavailable": [...],
                                "states": {eid: state}, "timers": false (no
                                voice device), "turns": [...]}]}
Timers are simulated (start/cancel/status by name), get_time returns the
current time, broadcasts always succeed, music_library returns the names of
"library": {media_type: [names]} (scenario or file level). "attrs": {eid: attributes} adds
attributes (weather: later_today/forecast in model format), "verbosity",
"location" and "player" (media player of the voice device) set the prompt.

A turn is a sentence or {"text": ..., "calls": [[tool, spec], ...],
"no_tool": true, "contains": [...]}: with expectations the turn is scored.
control spec: {"service", "entity_id" (or "area"), "data" (subset, case-
insensitive)}; get_state spec: {"entity_ids": [...]} (subset); other tools:
argument subset. Extra control calls fail the turn. The summary counts the
scored turns.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime
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
    def __init__(self, layout, unavailable=(), states=None, groups=()):
        self.layout = [tuple(e) for e in layout]
        self.groups = list(groups)
        self.names = {e: n for e, n, _ in layout}
        self.state = {e: DEFAULT_STATE.get(e.split(".")[0], "unknown") for e in self.names}
        self.state.update(states or {})
        for e in unavailable:
            self.state[e] = "unavailable"
        self.log: list[str] = []
        self.timers: list[dict] = []      # {"name", "seconds", "paused"}
        self.timer_device = True

    def timer(self, args):
        act = args.get("action")
        if act == "start" and args.get("command"):
            self.log.append(f"delayed command {args['command']!r}")
            return S.timer_result()
        if not self.timer_device:
            return S.timer_result("timers are not supported on this device")
        name = (args.get("name") or "").strip()
        secs = sum(int(args.get(k) or 0) * f for k, f in
                   (("hours", 3600), ("minutes", 60), ("seconds", 1)))

        def entries(ts):
            out = []
            for t in ts:
                m, s = divmod(t["seconds"], 60)
                h, m = divmod(m, 60)
                out.append(S.timer_status_entry(t["name"], h, m, s, not t["paused"]))
            return out
        if act == "start":
            if not secs:
                return S.timer_result("duration is missing")
            self.timers.append({"name": name, "seconds": secs, "paused": False})
            self.log.append(f"timer start {name or '-'} {secs}s")
            return S.timer_result()
        match = [t for t in self.timers if not name or t["name"].casefold() == name.casefold()]
        if act == "status":
            return S.timer_result(timers=entries(match))
        if act == "cancel_all":
            n = len(self.timers)
            self.timers = []
            return S.timer_result(canceled=n)
        if not match:
            return S.timer_result("no matching timer", entries(self.timers))
        if len(match) > 1:
            return S.timer_result("several timers match, name one", entries(match))
        t = match[0]
        if act == "cancel":
            self.timers.remove(t)
        elif act in ("pause", "resume"):
            t["paused"] = act == "pause"
        elif act in ("add", "subtract"):
            t["seconds"] = max(1, t["seconds"] + (secs if act == "add" else -secs))
        self.log.append(f"timer {act} {t['name'] or '-'}")
        return S.timer_result()

    attrs: dict = {}

    def state_ids(self, args):
        """entity_ids of a get_state call, as the integration resolves them:
        entity_ids, or every entity of `domain` in `area` ("*" = whole house),
        optionally only one device_class. None if the area is unknown."""
        ids = args.get("entity_ids") or []
        if isinstance(ids, str):
            ids = [ids]
        if not ids and (args.get("area") or args.get("domain")):
            ids = S.area_targets(args.get("domain") or "light", args.get("area") or S.ALL_AREAS,
                                 self.layout, self.groups)
            if ids is not None and args.get("device_class"):
                ids = [e for e in ids
                       if self.attrs.get(e, {}).get("device_class") == args["device_class"]]
        return ids

    def get_state(self, args):
        ids = self.state_ids(args)
        if ids is None:
            return S.dumps([{"error": f"unbekannter Bereich {args.get('area')}"}])
        out = []
        for eid in ids:
            if eid not in self.names:
                out.append({"entity_id": eid, "error": "unbekannte entity_id"})
            else:
                out.append(S.format_state(eid, self.names[eid], self.state[eid],
                                          self.attrs.get(eid, {})))
        return S.dumps(out)

    def music_library(self, args):
        eid = args.get("entity_id") or ""
        if eid not in self.names or not eid.startswith("media_player."):
            return S.library_result(error=f"Unknown entity {eid or '-'}")
        if self.state.get(eid) == "unavailable":
            return S.library_result(error=f"Entity {eid} is unavailable")
        names = self.library.get(args.get("media_type") or "playlist", [])
        q = str(args.get("search") or "").casefold()
        return S.library_result([n for n in names if q in n.casefold()])

    def control(self, args):
        errors, done = [], []
        by_area = False
        acts = args.get("actions") or []
        for act in acts if isinstance(acts, list) else [acts]:
            service = act.get("service", "")
            ids = act.get("entity_id") or []
            ids = [ids] if isinstance(ids, str) else ids
            if service not in S.SERVICES:
                errors.append(f"Unknown service {service}")
                continue
            domain, name = service.split(".", 1)
            if not ids and act.get("area"):
                if domain not in S.AREA_DOMAINS:
                    errors.append(f"area is not supported for {service}")
                    continue
                ids = S.area_targets(domain, act["area"], self.layout, self.groups)
                if ids is None:
                    errors.append(f"Unknown area {act['area']}")
                    continue
                if not ids:
                    errors.append(f"No {domain} entities in area {act['area']}")
                    continue
                by_area = True
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
        return S.control_result(errors, done if errors or by_area else None)


def norm(v):
    return v.strip().casefold() if isinstance(v, str) else v


def subset(spec, args):
    return all(norm(args.get(k)) == norm(v) if not isinstance(v, (list, dict))
               else (set(map(norm, v)) <= set(map(norm, args.get(k) or [])) if isinstance(v, list)
                     else subset(v, args.get(k) or {}))
               for k, v in spec.items())


def control_pairs(house, args):
    out = []
    acts = args.get("actions") or []
    for a in acts if isinstance(acts, list) else [acts]:
        dom = (a.get("service") or "").split(".", 1)[0]
        ids = a.get("entity_id") or []
        ids = [ids] if isinstance(ids, str) else ids
        if not ids and a.get("area"):
            ids = S.area_targets(dom, a["area"], house.layout, house.groups) or []
        out += [(a.get("service"), i, a.get("data") or {}) for i in ids]
    return out


def score(house, exp, calls):
    """calls: [(tool, args)] of the turn -> (ok, reason)."""
    if exp.get("no_tool"):
        return (not calls, "tool called" if calls else "")
    for tool, spec in exp.get("calls", []):
        if tool == "control":
            want = control_pairs(house, {"actions": [spec]})
            got = [p for t, a in calls if t == "control" for p in control_pairs(house, a)]
            for svc, eid, data in want:
                if not any(g[0] == svc and g[1] == eid and subset(data, g[2]) for g in got):
                    return False, f"missing {svc} {eid} {data}"
        elif tool == "get_state":
            got = set()
            for t, a in calls:
                if t == "get_state":
                    got |= set(house.state_ids(a) or [])
            if not set(spec.get("entity_ids", [])) <= got:
                return False, f"get_state missing {set(spec['entity_ids']) - got}"
        elif not any(t == tool and subset(spec, a) for t, a in calls):
            return False, f"missing {tool} {spec}"
    if any(t == "control" for t, _ in calls) and \
            not any(t == "control" for t, _ in exp.get("calls", [])):
        return False, "unexpected control"
    want = {(s, e) for t, sp in exp.get("calls", []) if t == "control"
            for s, e, _ in control_pairs(house, {"actions": [sp]})}
    extra = [(s, e) for t, a in calls if t == "control"
             for s, e, _ in control_pairs(house, a) if (s, e) not in want]
    if extra:
        return False, f"extra control {extra}"
    return True, ""


TOOLS = S.TOOLS


def complete(url, messages):
    body = json.dumps({"messages": messages, "tools": TOOLS, "temperature": 0,
                       "max_tokens": 400}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]


def run(url, layout, sc, groups=(), results=None, attrs=None):
    house = House(layout, sc.get("unavailable", []), sc.get("states"), groups)
    house.timer_device = sc.get("timers", True)
    house.attrs = {**(attrs or {}), **sc.get("attrs", {})}
    house.library = sc.get("library", {})
    system = S.build_system_prompt([tuple(e) for e in layout], sc.get("location"),
                                   verbosity=sc.get("verbosity", S.DEFAULT_VERBOSITY),
                                   player=sc.get("player"))
    history: list[dict] = []
    print(f"\n### {sc['name']}")
    for item in sc["turns"]:
        exp = item if isinstance(item, dict) else {}
        text = exp.get("text", item)
        turn = [{"role": "user", "content": text}]
        raw_calls = []
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
                if fn["name"] == "control":
                    res = house.control(args)
                elif fn["name"] == "get_state":
                    res = house.get_state(args)
                elif fn["name"] == "timer":
                    res = house.timer(args)
                elif fn["name"] == "music_library":
                    res = house.music_library(args)
                elif fn["name"] == "get_time":
                    res = S.time_result(datetime.now())
                elif fn["name"] == "broadcast":
                    res = S.dumps({"success": True})
                    house.log.append(f"broadcast {args.get('message')!r}")
                else:
                    res = S.dumps({"error": f"unknown tool {fn['name']}"})
                calls_seen.append(f"{fn['name']}({json.dumps(args, ensure_ascii=False)}) -> {res}")
                raw_calls.append((fn["name"], args))
                turn.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": res})
        history += turn
        print(f"  USER: {text}")
        for c in calls_seen:
            print(f"    tool: {c[:260]}")
        print(f"  BOT : {answer}")
        if exp and ("calls" in exp or "no_tool" in exp or "contains" in exp):
            ok, why = score(house, exp, raw_calls)
            if ok and exp.get("contains"):
                miss = [c for c in exp["contains"] if c.lower() not in (answer or "").lower()]
                ok, why = not miss, f"answer lacks {miss}" if miss else ""
            print(f"  {'PASS' if ok else 'FAIL ' + why}")
            if results is not None:
                results.append((sc["name"], text, ok, why))


def main():
    url = sys.argv[1].rstrip("/")
    cfg = json.load(open(sys.argv[2], encoding="utf-8"))
    base = Path(sys.argv[2]).parent
    results: list = []
    for sc in cfg["scenarios"]:
        layout = json.load(open(base / sc.get("house", cfg["house"]), encoding="utf-8"))
        states = cfg.get("states", {})
        if isinstance(states, str):
            states = json.load(open(base / states, encoding="utf-8"))
        sc = {**sc, "states": {**states, **sc.get("states", {})},
              "library": sc.get("library", cfg.get("library", {}))}
        if "verbosity" not in sc and "verbosity" in cfg:
            sc["verbosity"] = cfg["verbosity"]
        run(url, layout, sc, sc.get("groups", cfg.get("groups", [])), results,
            cfg.get("attrs"))
    if results:
        ok = sum(r[2] for r in results)
        print(f"\n== {ok}/{len(results)} scored turns passed")
        for name, text, good, why in results:
            if not good:
                print(f"  FAIL [{name}] {text!r}: {why}")


if __name__ == "__main__":
    main()
