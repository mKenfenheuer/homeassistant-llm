"""Custom frontend integration: German smart-home conversation agent.

Calls the llama.cpp OpenAI-compatible server (exposed by the ha-llm-inference
add-on via hass_services) and executes the model's tool calls against the
HA core. Uses only core primitives (verified on core 2026.10.x):
  hass.states.async_all(), hass.services.async_call(domain, service, data).

Tool names match the fine-tuned model's training schema (src/ha_config.py):
get_states / get_entity_attributes / control.
"""

import json
import os
from typing import Any

import aiohttp

DEFAULT_SYSTEM = (
    "You are a home assistant. Answer in German. You control the smart home with "
    "functions and answer questions about the state of the home. If the user "
    "mentions an entity that does not exist, say so and never make up an entity_id."
)
MODEL_NAME = "qwen2.5-3b-ha"
MAX_TOOL_ROUNDS = 4

DOMAIN_ACTIONS = {
    "light": ["light.turn_on", "light.turn_off",
              "light.set_brightness", "light.set_color"],
    "switch": ["switch.turn_on", "switch.turn_off"],
    "cover": ["cover.open", "cover.close", "cover.stop"],
    "alarm_control_panel": ["alarm_panel.arm_home", "alarm_panel.arm_away",
                            "alarm_panel.arm_night", "alarm_panel.arm_vacation",
                            "alarm_panel.disarm"],
    "thermostat": ["thermostat.set_temperature"],
}
# model action prefix -> HA service domain
ACTION_DOMAIN = {"alarm_panel": "alarm_control_panel"}


def server_url() -> str:
    try:
        services = json.loads(os.environ.get("hass_services", "{}"))
        url = services.get("ha_llm_inference")
    except (ValueError, TypeError):
        url = None
    return url or "http://localhost:8080/v1"


def states_dict() -> dict[str, Any]:
    from homeassistant.core import async_get_hass
    hass = async_get_hass()
    out = {}
    for s in hass.states.all():  # type: ignore
        out[s.entity_id] = {"state": s.state, **dict(s.attributes)}
    return out


def build_tools() -> list[dict]:
    eids = sorted(states_dict())
    actions: list[str] = []
    for e in eids:
        actions += DOMAIN_ACTIONS.get(e.split(".", 1)[0], [])
    return [
        {"type": "function", "function": {
            "name": "get_states",
            "description": "Current state of all entities.",
            "parameters": {"type": "object", "properties": {}}}},
        {"type": "function", "function": {
            "name": "get_entity_attributes",
            "description": "Attributes and state of one entity.",
            "parameters": {"type": "object",
                           "properties": {"entity_id": {"type": "string"}},
                           "required": ["entity_id"]}}},
        {"type": "function", "function": {
            "name": "control",
            "description": "Control entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string",
                                           "enum": sorted(set(actions))},
                                "entity_id": {"type": "string", "enum": eids},
                                "data": {"type": "object"},
                            },
                        },
                    },
                },
                "required": ["actions"],
            },
        }},
    ]


async def _run_action(act: dict) -> dict:
    try:
        prefix, cmd = act["action"].split(".", 1)
        domain = ACTION_DOMAIN.get(prefix, prefix)
        data = {"entity_id": act["entity_id"], **(act.get("data") or {})}
        from homeassistant.core import async_get_hass
        hass = async_get_hass()
        await hass.services.async_call(domain, cmd, data,
                                       blocking=False, return_response=False)
        return {"ok": True, "entity_id": act["entity_id"]}
    except Exception as err:  # noqa
        return {"ok": False, "error": str(err), "entity_id": act.get("entity_id")}


async def async_process_request(*args) -> dict[str, Any] | None:
    # tolerate both call shapes (ConversationInputData or plain list)
    req = args[0]
    if isinstance(req, dict) and "messages" in req:
        messages = list(req["messages"])
    elif isinstance(req, list):
        messages = list(req)
    else:
        try:
            messages = list(req.messages)  # type: ignore
        except Exception:
            return None
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": DEFAULT_SYSTEM}] + messages

    url = server_url() + "/chat/completions"
    rounds = 0
    while True:
        body = {
            "model": MODEL_NAME,
            "messages": messages,
            "tools": build_tools(),
            "tool_choice": "auto",
            "max_tokens": 512,
            "temperature": 0.2,
        }
        async with aiohttp.ClientSession() as session, \
                session.post(url, json=body,
                             timeout=aiohttp.ClientTimeout(total=120)) as resp:
            data = await resp.json()
        msg = data["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant",
                         **{k: msg[k] for k in ("content", "tool_calls")
                            if msg.get(k)}})
        rounds += 1
        if not calls or rounds > MAX_TOOL_ROUNDS:
            response = (msg.get("content") or "").strip()
            return {"response": response or "Something went wrong.",
                    "conversation": messages}
        for tc in calls:
            name = tc["function"]["name"]
            try:
                fnargs = json.loads(tc["function"]["arguments"])
            except (ValueError, TypeError):
                fnargs = {}
            if name == "get_states":
                result = states_dict()
            elif name == "get_entity_attributes":
                result = {fnargs.get("entity_id"):
                          states_dict().get(fnargs.get("entity_id"), {})}
            elif name == "control":
                results = [await _run_action(a)
                           for a in fnargs.get("actions", [])]
                result = {"ok": all(r.get("ok") for r in results),
                          "results": results}
            else:
                result = {"error": f"unknown tool {name}"}
            messages.append({"role": "tool", "tool_call_id": tc["id"],
                             "content": json.dumps(result, ensure_ascii=False)})
