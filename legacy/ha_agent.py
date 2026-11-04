#!/usr/bin/env python3
"""Runtime: chat with the fine-tuned model, wired to your real Home Assistant.

Talks to a llama.cpp server (OpenAI-compatible /v1/chat/completions with tools)
and executes tool calls against the HA REST API. Tool schemas are built from
YOUR live entity list, so the fine-tuned model transfers to any house.

Env:
  HA_URL   e.g. http://10.0.0.1:8123
  HA_TOKEN long-lived HA token
  LLM_URL  e.g. http://localhost:8080/v1  (llama-server --port 8080)
Usage:
  ha_agent.py "schlight the kitchen lights"
  ha_agent.py            # interactive loop
"""

import json
import os
import sys
import urllib.request

HA_URL = os.environ.get("HA_URL", "http://127.0.0.1:8123")
HA_TOKEN = os.environ.get("HA_TOKEN", "")
LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:8080/v1")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5-3b-ha")

SYSTEM = (
    "You are a home assistant. Answer in German. You control the smart home with "
    "functions and answer questions about the state of the home. Never invent an "
    "entity_id; if an entity is unknown, say so."
)


def ha_get(path):
    return _ha(path, None)


def ha_post(path, body):
    return _ha(path, body)


def _ha(path, body=None):
    req = urllib.request.Request(
        HA_URL + path, method="POST" if body is not None else "GET",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": "Bearer " + HA_TOKEN,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


def build_tools():
    """Tool schemas with entity enums from the live HA state."""
    states = ha_get("/api/states")
    eids = list(states)
    control_actions = set()
    for e in eids:
        d = e.split(".")[0]
        acts = {
            "light": ["light.turn_on", "light.turn_off",
                      "light.set_brightness", "light.set_color"],
            "switch": ["switch.turn_on", "switch.turn_off"],
            "cover": ["cover.open", "cover.close", "cover.stop"],
            "alarm_control_panel": ["alarm_panel.arm_home",
                                    "alarm_panel.arm_away",
                                    "alarm_panel.arm_night",
                                    "alarm_panel.arm_vacation",
                                    "alarm_panel.disarm"],
            "thermostat": ["thermostat.set_temperature"],
        }.get(d, [])
        control_actions.update(acts)
    return [
        {"type": "function", "function": {
            "name": "get_states",
            "description": "Current state of all entities.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "control",
            "description": "Control entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {"type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string",
                                       "enum": sorted(control_actions)},
                            "entity_id": {"type": "string", "enum": eids},
                            "data": {"type": "object"}}}}},
                "required": ["actions"]}}},
    ]


def run_tool(name, args):
    if name == "get_states":
        return json.dumps(ha_get("/api/states"), ensure_ascii=False)
    if name == "control":
        out = []
        for a in args["actions"]:
            domain, action = a["action"].split(".")
            body = {"entity_id": a["entity_id"], **(a.get("data") or {})}
            try:
                ha_post("/api/services/%s/%s" % (domain, action), body)
                out.append({"ok": True, "call": a})
            except Exception as e:
                out.append({"ok": False, "error": str(e), "call": a})
        return json.dumps(out, ensure_ascii=False)
    return json.dumps({"error": "unknown tool"})


def chat_loop(history):
    tools = build_tools()
    msgs = [{"role": "system", "content": SYSTEM}] + history
    for _ in range(4):
        body = json.dumps({"model": MODEL, "messages": msgs, "tools": tools,
                           "max_tokens": 512, "temperature": 0.2})
        req = urllib.request.Request(
            LLM_URL + "/chat/completions", data=body.encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            msg = json.loads(r.read())["choices"][0]["message"]
        msgs.append({"role": "assistant", **{k: msg[k] for k in
                                          ("content", "tool_calls") if msg.get(k)}})
        if not msg.get("tool_calls"):
            return msgs
        for tc in msg["tool_calls"]:
            res = run_tool(tc["function"]["name"],
                           json.loads(tc["function"]["arguments"]))
            msgs.append({"role": "tool", "tool_call_id": tc["id"],
                         "content": res})


def main():
    args = sys.argv[1:]
    if args:
        history = [{"role": "user", "content": " ".join(args)}]
    else:
        history = []
        print("type 'exit' to quit")
        while True:
            try:
                t = input("> ")
            except EOFError:
                break
            if t.strip() in ("exit", "quit"):
                break
            history.append({"role": "user", "content": t})
            msgs = chat_loop(history)
            print((msgs[-1].get("content") or "").strip())
            history = msgs[1:]
        return
    msgs = chat_loop(history)
    for m in msgs[1:]:
        if m["role"] == "assistant" and m.get("content"):
            print(m["content"])


if __name__ == "__main__":
    main()
