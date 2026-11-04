"""Conversation agent backed by the fine-tuned model on llama-server."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import OrderedDict
from datetime import timedelta
from typing import Any, Literal

import aiohttp

from homeassistant.components import conversation
from homeassistant.components.homeassistant.exposed_entities import (
    async_should_expose,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import MATCH_ALL, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import (
    area_registry as ar,
    device_registry as dr,
    entity_registry as er,
    intent,
    issue_registry as ir,
)
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.event import async_track_time_interval
from homeassistant.helpers.start import async_at_started

from . import llm_schema as S
from .const import (
    CONF_MAX_ENTITIES,
    CONF_MAX_HISTORY,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_URL,
    CONF_USE_LOCATION,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_HISTORY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_USE_LOCATION,
    DOMAIN,
    MAX_CONVERSATIONS,
    MAX_TOOL_ROUNDS,
    SLOW_WARMUP_S,
    WARMUP_INTERVAL_MIN,
    WARMUP_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry,
                            async_add_entities: AddConfigEntryEntitiesCallback
                            ) -> None:
    async_add_entities([HaLlmAgentEntity(entry)])


class HaLlmAgentEntity(conversation.ConversationEntity,
                       conversation.AbstractConversationAgent):
    """German smart-home agent using a local llama.cpp server."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_supported_features = conversation.ConversationEntityFeature.CONTROL

    def __init__(self, entry: ConfigEntry) -> None:
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)}, name=entry.title,
            manufacturer="llama.cpp", model="Qwen2.5-3B HA (fine-tuned)",
            entry_type=DeviceEntryType.SERVICE)
        # conversation_id -> OpenAI-format messages without system prompt
        self._history: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        # hash of the system prompt that llama-server has in its cache
        self._cached_prompt: str | None = None
        self._warming = False

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()
        conversation.async_set_agent(self.hass, self.entry, self)
        # prefill llama-server's prompt cache before the first question
        self.async_on_remove(async_at_started(self.hass, self._async_start_warmup))
        self.async_on_remove(async_track_time_interval(
            self.hass, self._async_warmup_tick, timedelta(minutes=WARMUP_INTERVAL_MIN)))

    async def _async_start_warmup(self, _hass) -> None:
        self.entry.async_create_background_task(
            self.hass, self._async_warmup(), "ha_llm_agent warmup")

    async def _async_warmup_tick(self, _now) -> None:
        await self._async_start_warmup(self.hass)

    async def _async_warmup(self) -> None:
        """Send the current system prompt once so llama-server caches it."""
        if self._warming:
            return
        exposed = self._exposed()
        system = S.build_system_prompt(
            [(eid, name, area) for eid, (name, area) in exposed.items()])
        digest = hashlib.sha1(system.encode()).hexdigest()
        if digest == self._cached_prompt:
            return
        self._warming = True
        start = time.monotonic()
        try:
            await self._complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": "Hallo"}],
                max_tokens=1, timeout=WARMUP_TIMEOUT)
            self._cached_prompt = digest
            took = time.monotonic() - start
            if took > SLOW_WARMUP_S:
                _LOGGER.warning(
                    "Prompt cache warm-up for %d entities took %.0f s. The CPU of "
                    "the llama-server host is slow for this prompt size (no AVX2?); "
                    "expose fewer entities or run llama-server on a faster machine",
                    len(exposed), took)
                ir.async_create_issue(
                    self.hass, DOMAIN, f"slow_inference_{self.entry.entry_id}",
                    is_fixable=False, severity=ir.IssueSeverity.WARNING,
                    translation_key="slow_inference",
                    translation_placeholders={
                        "seconds": f"{took:.0f}", "entities": str(len(exposed)),
                        "url": self.entry.data[CONF_URL]})
            else:
                _LOGGER.info("Prompt cache warmed (%d entities) in %.0f s",
                             len(exposed), took)
                ir.async_delete_issue(
                    self.hass, DOMAIN, f"slow_inference_{self.entry.entry_id}")
        except (aiohttp.ClientError, asyncio.TimeoutError, HomeAssistantError,
                KeyError, ValueError) as err:
            _LOGGER.warning("Prompt cache warm-up failed after %.0f s: %s: %s",
                            time.monotonic() - start, type(err).__name__, err)
        finally:
            self._warming = False

    async def async_will_remove_from_hass(self) -> None:
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    # ------------------------------------------------------------------ opts
    def _opt(self, key, default):
        return self.entry.options.get(key, default)

    # -------------------------------------------------------------- entities
    def _exposed(self) -> dict[str, tuple[str, str | None]]:
        """entity_id -> (name, area) for all entities exposed to Assist."""
        ent_reg = er.async_get(self.hass)
        dev_reg = dr.async_get(self.hass)
        area_reg = ar.async_get(self.hass)
        out: dict[str, tuple[str, str | None]] = {}
        for state in self.hass.states.async_all():
            if state.domain not in S.EXPOSED_DOMAINS:
                continue
            if not async_should_expose(self.hass, conversation.DOMAIN,
                                       state.entity_id):
                continue
            area_id = None
            if (entry := ent_reg.async_get(state.entity_id)) is not None:
                area_id = entry.area_id
                if area_id is None and entry.device_id and (
                        dev := dev_reg.async_get(entry.device_id)):
                    area_id = dev.area_id
            area = area_reg.async_get_area(area_id) if area_id else None
            out[state.entity_id] = (state.name, area.name if area else None)
        limit = int(self._opt(CONF_MAX_ENTITIES, DEFAULT_MAX_ENTITIES))
        if len(out) > limit:
            _LOGGER.warning("%d entities exposed, only the first %d are sent "
                            "to the model", len(out), limit)
            # controllable devices first, read-only sensors last
            keep = sorted(out, key=lambda e: (e.split(".")[0] in S.READ_DOMAINS, e))
            out = {e: out[e] for e in keep[:limit]}
        return out

    def _location(self, user_input: conversation.ConversationInput) -> str | None:
        area_id = None
        if user_input.device_id and (
                dev := dr.async_get(self.hass).async_get(user_input.device_id)):
            area_id = dev.area_id
        if area_id is None and user_input.satellite_id and (
                ent := er.async_get(self.hass).async_get(user_input.satellite_id)):
            area_id = ent.area_id
        area = ar.async_get(self.hass).async_get_area(area_id) if area_id else None
        return area.name if area else None

    # ----------------------------------------------------------------- tools
    async def _get_state(self, args: dict, exposed: dict) -> str:
        ids = args.get("entity_ids") or args.get("entity_id") or []
        if isinstance(ids, str):
            ids = [ids]
        out = []
        for eid in ids[:50]:
            state = self.hass.states.get(eid)
            if eid not in exposed or state is None:
                out.append({"entity_id": eid, "error": "unbekannte entity_id"})
                continue
            out.append(S.format_state(eid, exposed[eid][0], state.state,
                                      dict(state.attributes)))
        return S.dumps(out)

    async def _control(self, args: dict, exposed: dict,
                       context) -> str:
        errors: list[str] = []
        actions = args.get("actions")
        if isinstance(actions, dict):
            actions = [actions]
        for act in actions or []:
            service = act.get("service", "")
            ids = act.get("entity_id") or []
            if isinstance(ids, str):
                ids = [ids]
            if service not in S.SERVICES:
                errors.append(f"Unknown service {service}")
                continue
            domain, name = service.split(".", 1)
            bad = [i for i in ids if i not in exposed or i.split(".")[0] != domain]
            if bad or not ids:
                errors.append(f"Unknown entity for {service}: {', '.join(bad) or '-'}")
                continue
            unavailable = [i for i in ids if (s := self.hass.states.get(i)) is None
                           or s.state == STATE_UNAVAILABLE]
            if unavailable:
                errors.append(f"Entity {', '.join(unavailable)} is unavailable")
                continue
            data = {k: v for k, v in (act.get("data") or {}).items()
                    if k in S.SERVICES[service]}
            try:
                await self.hass.services.async_call(
                    domain, name, {"entity_id": ids, **data},
                    blocking=True, context=context)
            except (HomeAssistantError, ValueError, TypeError) as err:
                errors.append(f"{service} failed for {', '.join(ids)}: {err}")
        return S.control_result(errors)

    # ------------------------------------------------------------------- llm
    async def _complete(self, messages: list[dict], max_tokens: int = 400,
                        timeout: float | None = None) -> dict:
        url = self.entry.data[CONF_URL].rstrip("/") + "/v1/chat/completions"
        body = {
            "model": "ha-llm",
            "messages": messages,
            "tools": S.TOOLS,
            "temperature": float(self._opt(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)),
            "max_tokens": max_tokens,
            "cache_prompt": True,
        }
        session = async_get_clientsession(self.hass)
        timeout = aiohttp.ClientTimeout(
            total=timeout or float(self._opt(CONF_TIMEOUT, DEFAULT_TIMEOUT)))
        async with session.post(url, json=body, timeout=timeout) as resp:
            if resp.status != 200:
                raise HomeAssistantError(
                    f"llama-server HTTP {resp.status}: {(await resp.text())[:200]}")
            data = await resp.json()
        msg = data["choices"][0]["message"]
        # fallback: server did not parse the Qwen <tool_call> block
        if not msg.get("tool_calls") and "<tool_call>" in (msg.get("content") or ""):
            calls = []
            for i, raw in enumerate(_TOOL_CALL_RE.findall(msg["content"])):
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                calls.append({"id": f"call_{i}", "type": "function", "function": {
                    "name": obj.get("name"),
                    "arguments": S.dumps(obj.get("arguments", {}))}})
            msg = {"role": "assistant", "content": None, "tool_calls": calls}
        return msg

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        conv_id = chat_log.conversation_id
        exposed = self._exposed()
        location = (self._location(user_input)
                    if self._opt(CONF_USE_LOCATION, DEFAULT_USE_LOCATION) else None)
        system = S.build_system_prompt(
            [(eid, name, area) for eid, (name, area) in exposed.items()], location)
        history = self._history.get(conv_id, [])
        turn: list[dict[str, Any]] = [{"role": "user", "content": user_input.text}]
        text = ""
        try:
            for _ in range(MAX_TOOL_ROUNDS + 1):
                msg = await self._complete(
                    [{"role": "system", "content": system}] + history + turn)
                calls = msg.get("tool_calls") or []
                if not calls:
                    text = (msg.get("content") or "").strip()
                    turn.append({"role": "assistant", "content": text})
                    break
                turn.append({"role": "assistant", "content": None,
                             "tool_calls": calls})
                for call in calls:
                    fn = call.get("function", {})
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    if fn.get("name") == "control":
                        result = await self._control(args, exposed,
                                                     user_input.context)
                    elif fn.get("name") == "get_state":
                        result = await self._get_state(args, exposed)
                    else:
                        result = S.dumps({"error": f"unknown tool {fn.get('name')}"})
                    _LOGGER.debug("tool %s(%s) -> %s", fn.get("name"), args, result)
                    turn.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": result})
            else:
                text = "Das konnte ich leider nicht vollständig ausführen."
        except (aiohttp.ClientError, asyncio.TimeoutError, HomeAssistantError,
                KeyError, ValueError) as err:
            _LOGGER.error("LLM request failed: %s: %s", type(err).__name__, err)
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                "Der Sprachassistent hat nicht rechtzeitig geantwortet."
                if isinstance(err, asyncio.TimeoutError)
                else "Der Sprachassistent ist gerade nicht erreichbar.")
            return conversation.ConversationResult(
                response=response, conversation_id=conv_id)

        if location is None:
            self._cached_prompt = hashlib.sha1(system.encode()).hexdigest()

        # keep the last N user turns for follow-up questions
        new_hist = history + turn
        max_turns = int(self._opt(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY))
        user_idx = [i for i, m in enumerate(new_hist) if m["role"] == "user"]
        if max_turns <= 0:
            new_hist = []
        elif len(user_idx) > max_turns:
            new_hist = new_hist[user_idx[-max_turns]:]
        self._history[conv_id] = new_hist
        self._history.move_to_end(conv_id)
        while len(self._history) > MAX_CONVERSATIONS:
            self._history.popitem(last=False)

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=self.entity_id,
                                          content=text or "Erledigt."))
        return conversation.async_get_result_from_chat_log(user_input, chat_log)
