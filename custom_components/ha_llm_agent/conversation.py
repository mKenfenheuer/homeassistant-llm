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
from homeassistant.util import dt as dt_util

from . import llm_schema as S
from .const import (
    CONF_LANGUAGE,
    CONF_VERBOSITY,
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
    """Smart-home agent using a local llama.cpp server (one model per language)."""

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
            [(eid, name, area) for eid, (name, area) in exposed.items()], lang=self._lang,
            verbosity=self._verbosity)
        digest = hashlib.sha1(system.encode()).hexdigest()
        if digest == self._cached_prompt:
            return
        self._warming = True
        start = time.monotonic()
        try:
            await self._complete(
                [{"role": "system", "content": system},
                 {"role": "user", "content": S.lang_texts(self._lang)["warmup_text"]}],
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

    @property
    def _verbosity(self) -> str:
        v = self._opt(CONF_VERBOSITY, S.DEFAULT_VERBOSITY)
        return v if v in S.VERBOSITY else S.DEFAULT_VERBOSITY

    @property
    def _lang(self) -> str:
        """Language of the model: option, else the Home Assistant language."""
        return S.pick_language(self._opt(CONF_LANGUAGE, None)
                               or self.hass.config.language)

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
    def _area_ids(self, domain: str, area: str, exposed: dict) -> list[str] | None:
        """Exposed entities of `domain` in `area` ("*" = whole house), as the
        device list shows them. Group entities (state attribute entity_id)
        are left out, their members are included."""
        groups = [eid for eid in exposed
                  if (st := self.hass.states.get(eid)) is not None
                  and isinstance(st.attributes.get("entity_id"), list)]
        return S.area_targets(domain, str(area),
                              [(e, n, a) for e, (n, a) in exposed.items()], groups)

    async def _get_state(self, args: dict, exposed: dict) -> str:
        t = S.lang_texts(self._lang)
        ids = args.get("entity_ids") or args.get("entity_id") or []
        if isinstance(ids, str):
            ids = [ids]
        if not ids and args.get("area"):
            domain = args.get("domain") or "light"
            if domain not in S.AREA_DOMAINS:
                return S.dumps([{"error": t["error_area_domain"].format(domain=domain)}])
            ids = self._area_ids(domain, args["area"], exposed)
            if ids is None:
                return S.dumps([{"error": t["error_area_unknown"].format(area=args["area"])}])
        out = []
        for eid in ids[:100]:
            state = self.hass.states.get(eid)
            if eid not in exposed or state is None:
                out.append({"entity_id": eid, "error": t["error_entity_unknown"]})
                continue
            attrs = dict(state.attributes)
            if state.state != STATE_UNAVAILABLE:
                if state.domain == "weather":
                    attrs["later_today"] = await self._hourly(eid)
                    attrs["forecast"] = await self._forecast(eid)
                elif state.domain == "todo":
                    attrs["items"] = await self._todo_items(eid)
            out.append(S.format_state(eid, exposed[eid][0], state.state, attrs,
                                      self._lang))
        return S.dumps(out)

    async def _forecast(self, eid: str) -> list[dict] | None:
        """Daily forecast of a weather entity (twice daily: day parts only)."""
        for kind in ("daily", "twice_daily"):
            try:
                resp = await self.hass.services.async_call(
                    "weather", "get_forecasts", {"entity_id": eid, "type": kind},
                    blocking=True, return_response=True)
            except (HomeAssistantError, ValueError) as err:
                _LOGGER.debug("No %s forecast for %s: %s", kind, eid, err)
                continue
            entries = ((resp or {}).get(eid) or {}).get("forecast") or []
            if kind == "twice_daily":
                entries = [f for f in entries if f.get("is_daytime", True)]
            if entries:
                return S.format_forecast(entries, dt_util.now().date(), self._lang) or None
        return None

    async def _hourly(self, eid: str) -> list[dict] | None:
        """Rest of today from the hourly forecast."""
        try:
            resp = await self.hass.services.async_call(
                "weather", "get_forecasts", {"entity_id": eid, "type": "hourly"},
                blocking=True, return_response=True)
        except (HomeAssistantError, ValueError) as err:
            _LOGGER.debug("No hourly forecast for %s: %s", eid, err)
            return None
        entries = ((resp or {}).get(eid) or {}).get("forecast") or []
        return S.format_hourly(entries, dt_util.now(), self._lang) or None

    async def _todo_items(self, eid: str) -> list[str]:
        """Open items of a to-do list."""
        try:
            resp = await self.hass.services.async_call(
                "todo", "get_items", {"entity_id": eid, "status": ["needs_action"]},
                blocking=True, return_response=True)
        except (HomeAssistantError, ValueError) as err:
            _LOGGER.debug("Could not read %s: %s", eid, err)
            return []
        items = ((resp or {}).get(eid) or {}).get("items") or []
        return [i.get("summary", "") for i in items][:S.TODO_ITEMS]

    async def _control(self, args: dict, exposed: dict,
                       context) -> str:
        errors: list[str] = []
        done: list[str] = []
        by_area = False
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
            if not ids and act.get("area"):
                if domain not in S.AREA_DOMAINS:
                    errors.append(f"area is not supported for {service}")
                    continue
                ids = self._area_ids(domain, act["area"], exposed)
                if ids is None:
                    errors.append(f"Unknown area {act['area']}")
                    continue
                if not ids:
                    errors.append(f"No {domain} entities in area {act['area']}")
                    continue
                by_area = True
            bad = [i for i in ids if i not in exposed or i.split(".")[0] != domain]
            if bad or not ids:
                errors.append(f"Unknown entity for {service}: {', '.join(bad) or '-'}")
                continue
            unavailable = [i for i in ids if (s := self.hass.states.get(i)) is None
                           or s.state == STATE_UNAVAILABLE]
            if unavailable:
                # switch the reachable entities, report the others as failed
                errors.append(f"Entity {', '.join(unavailable)} is unavailable")
                ids = [i for i in ids if i not in unavailable]
                if not ids:
                    continue
            data = {k: v for k, v in (act.get("data") or {}).items()
                    if k in S.SERVICES[service]}
            try:
                if service == "media_player.search_and_play":
                    for eid in ids:
                        await self._search_and_play(eid, data, context)
                elif service == "todo.complete_item":
                    await self.hass.services.async_call(
                        "todo", "update_item",
                        {"entity_id": ids, "item": data.get("item", ""),
                         "status": "completed"}, blocking=True, context=context)
                else:
                    await self.hass.services.async_call(
                        domain, name, {"entity_id": ids, **data},
                        blocking=True, context=context)
            except (HomeAssistantError, ValueError, TypeError) as err:
                errors.append(f"{service} failed for {', '.join(ids)}: {err}")
            else:
                done.extend(ids)
        # on success `done` is only reported for area actions: it tells the
        # model which entities the area stood for
        return S.control_result(errors, done if errors or by_area else None)

    async def _search_and_play(self, eid: str, data: dict, context) -> None:
        """Find music by name and play it: Music Assistant players through
        music_assistant.play_media, other players through search_media."""
        query = str(data.get("query") or "").strip()
        if not query:
            raise HomeAssistantError("query is missing")
        media_type = data.get("media_type")
        artist = str(data.get("artist") or "").strip()
        entry = er.async_get(self.hass).async_get(eid)
        if entry is not None and entry.platform == "music_assistant":
            svc: dict[str, Any] = {"entity_id": eid, "media_id": query}
            if media_type in S.MEDIA_TYPES:
                svc["media_type"] = media_type
            if artist and media_type != "artist":
                svc["artist"] = artist
            await self.hass.services.async_call(
                "music_assistant", "play_media", svc, blocking=True, context=context)
            return
        search: dict[str, Any] = {"entity_id": eid,
                                  "search_query": f"{query} {artist}".strip()}
        if media_type in ("track", "album", "artist", "playlist", "podcast"):
            search["media_filter_classes"] = [media_type]
        try:
            resp = await self.hass.services.async_call(
                "media_player", "search_media", search, blocking=True,
                return_response=True, context=context)
        except HomeAssistantError as err:
            raise HomeAssistantError(f"music search is not supported by {eid}") from err
        found = (resp or {}).get(eid)
        results = getattr(found, "result", None)
        if results is None and isinstance(found, dict):
            results = found.get("result")
        if not results:
            raise HomeAssistantError(f"No results found for {query}")
        first = results[0]
        get = first.get if isinstance(first, dict) else lambda k: getattr(first, k)
        await self.hass.services.async_call(
            "media_player", "play_media",
            {"entity_id": eid, "media_content_id": get("media_content_id"),
             "media_content_type": get("media_content_type")},
            blocking=True, context=context)

    async def _intent(self, intent_type: str, slots: dict[str, Any],
                      user_input: conversation.ConversationInput) -> intent.IntentResponse:
        return await intent.async_handle(
            self.hass, DOMAIN, intent_type,
            {k: {"value": v} for k, v in slots.items()},
            text_input=user_input.text, context=user_input.context,
            language=user_input.language, assistant=conversation.DOMAIN,
            device_id=user_input.device_id, satellite_id=user_input.satellite_id,
            conversation_agent_id=self.entity_id)

    async def _timer_list(self, user_input) -> list[dict]:
        try:
            resp = await self._intent(intent.INTENT_TIMER_STATUS, {}, user_input)
        except intent.IntentError:
            return []
        return [S.timer_status_entry(
            t.get("name", ""), t.get("rounded_hours_left", 0),
            t.get("rounded_minutes_left", 0), t.get("rounded_seconds_left", 0),
            t.get("is_active", True), self._lang)
            for t in resp.speech_slots.get("timers", [])]

    async def _timer(self, args: dict, user_input) -> str:
        """Timers of the voice device through the built-in timer intents."""
        action = args.get("action")
        intents = {"start": intent.INTENT_START_TIMER, "cancel": intent.INTENT_CANCEL_TIMER,
                   "cancel_all": intent.INTENT_CANCEL_ALL_TIMERS,
                   "pause": intent.INTENT_PAUSE_TIMER, "resume": intent.INTENT_UNPAUSE_TIMER,
                   "add": intent.INTENT_INCREASE_TIMER, "subtract": intent.INTENT_DECREASE_TIMER,
                   "status": intent.INTENT_TIMER_STATUS}
        if action not in intents:
            return S.timer_result(f"unknown action {action}")
        slots: dict[str, Any] = {}
        if action in ("start", "add", "subtract"):
            for key in ("hours", "minutes", "seconds"):
                try:
                    val = int(args.get(key) or 0)
                except (TypeError, ValueError):
                    val = 0
                if val > 0:
                    slots[key] = val
            if not slots:
                return S.timer_result("duration is missing")
        if args.get("name") and action != "cancel_all":
            slots["name"] = str(args["name"])
        if action == "start" and args.get("command"):
            slots["conversation_command"] = str(args["command"])
        try:
            resp = await self._intent(intents[action], slots, user_input)
        except intent.IntentError as err:
            kind = type(err).__name__
            if kind == "TimersNotSupportedError":
                return S.timer_result("timers are not supported on this device")
            if kind == "TimerNotFoundError":
                return S.timer_result("no matching timer", await self._timer_list(user_input))
            if kind == "MultipleTimersMatchedError":
                return S.timer_result("several timers match, name one",
                                      await self._timer_list(user_input))
            return S.timer_result(str(err) or kind)
        if action == "status":
            timers = [S.timer_status_entry(
                t.get("name", ""), t.get("rounded_hours_left", 0),
                t.get("rounded_minutes_left", 0), t.get("rounded_seconds_left", 0),
                t.get("is_active", True), self._lang)
                for t in resp.speech_slots.get("timers", [])]
            return S.timer_result(timers=timers)
        if action == "cancel_all":
            return S.timer_result(canceled=resp.speech_slots.get("canceled", 0))
        return S.timer_result()

    async def _broadcast(self, args: dict, user_input) -> str:
        message = str(args.get("message") or "").strip()
        if not message:
            return S.dumps({"success": False, "error": "message is missing"})
        try:
            resp = await self._intent(intent.INTENT_BROADCAST, {"message": message},
                                      user_input)
        except (intent.IntentError, HomeAssistantError) as err:
            return S.dumps({"success": False, "error": str(err) or type(err).__name__})
        if not resp.success_results:
            return S.dumps({"success": False,
                            "error": "no other voice devices for a broadcast"})
        return S.dumps({"success": True})

    # ------------------------------------------------------------------- llm
    async def _complete(self, messages: list[dict], max_tokens: int = 400,
                        timeout: float | None = None) -> dict:
        url = self.entry.data[CONF_URL].rstrip("/") + "/v1/chat/completions"
        body = {
            "model": "ha-llm",
            "messages": messages,
            "tools": S.build_tools(self._lang),
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
            [(eid, name, area) for eid, (name, area) in exposed.items()], location,
            lang=self._lang, verbosity=self._verbosity)
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
                    elif fn.get("name") == "timer":
                        result = await self._timer(args, user_input)
                    elif fn.get("name") == "get_time":
                        result = S.time_result(dt_util.now(), self._lang)
                    elif fn.get("name") == "broadcast":
                        result = await self._broadcast(args, user_input)
                    else:
                        result = S.dumps({"error": f"unknown tool {fn.get('name')}"})
                    _LOGGER.debug("tool %s(%s) -> %s", fn.get("name"), args, result)
                    turn.append({"role": "tool", "tool_call_id": call.get("id", ""),
                                 "content": result})
            else:
                text = S.lang_texts(self._lang)["error_incomplete"]
        except (aiohttp.ClientError, asyncio.TimeoutError, HomeAssistantError,
                KeyError, ValueError) as err:
            _LOGGER.error("LLM request failed: %s: %s", type(err).__name__, err)
            response = intent.IntentResponse(language=user_input.language)
            response.async_set_error(
                intent.IntentResponseErrorCode.UNKNOWN,
                S.lang_texts(self._lang)["error_timeout"]
                if isinstance(err, asyncio.TimeoutError)
                else S.lang_texts(self._lang)["error_unreachable"])
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
                                          content=text))
        return conversation.async_get_result_from_chat_log(user_input, chat_log)
