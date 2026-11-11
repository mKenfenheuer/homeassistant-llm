"""Config flow for HA LLM Agent."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp

try:  # Home Assistant >= 2026.x
    import probatio as vol
except ImportError:  # older cores
    import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.selector import (
    BooleanSelector,
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    ADDON_SUFFIXES,
    CONF_MAX_ENTITIES,
    CONF_TYPE,
    CONF_MAX_HISTORY,
    CONF_TEMPERATURE,
    CONF_TIMEOUT,
    CONF_URL,
    CONF_USE_LOCATION,
    DEFAULT_MAX_ENTITIES,
    DEFAULT_MAX_HISTORY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_URL,
    DEFAULT_USE_LOCATION,
    DEFAULT_WHISPER_URL,
    DOMAIN,
    TYPE_CONVERSATION,
    TYPE_STT,
    WHISPER_ADDON_SUFFIXES,
)


def _default_url(hass, kind: str = TYPE_CONVERSATION) -> str:
    """URL of the installed app of this kind, if any.

    A local app has the slug local_<name>, one from a git repository
    <hash>_<name>. The hostname is the slug with dashes. Earlier suffixes in
    the list are preferred.
    """
    suffixes, default = ((ADDON_SUFFIXES, DEFAULT_URL) if kind == TYPE_CONVERSATION
                         else (WHISPER_ADDON_SUFFIXES, DEFAULT_WHISPER_URL))
    try:
        from homeassistant.helpers.hassio import is_hassio

        if not is_hassio(hass):
            return default
        from homeassistant.components.hassio import get_addons_info

        found: dict[int, str] = {}
        for slug, info in (get_addons_info(hass) or {}).items():
            for rank, suffix in enumerate(suffixes):
                if slug.endswith(suffix) and rank not in found:
                    host = (info or {}).get("hostname") or slug.replace("_", "-")
                    found[rank] = f"http://{host}:8080"
        if found:
            return found[min(found)]
    except Exception:  # noqa: BLE001 - Supervisor not ready or API changed
        pass
    return default


async def _probe(hass, url: str, kind: str = TYPE_CONVERSATION) -> str | None:
    """Return an error key or None if the server answers.

    llama-server on /health, whisper-server on its start page.
    """
    session = async_get_clientsession(hass)
    path = "/health" if kind == TYPE_CONVERSATION else "/"
    try:
        async with session.get(url.rstrip("/") + path,
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            # 503 = server is up but still loading the model
            if resp.status in (200, 503):
                return None
            return "cannot_connect"
    except (aiohttp.ClientError, asyncio.TimeoutError):
        return "cannot_connect"


class HaLlmAgentConfigFlow(ConfigFlow, domain=DOMAIN):
    VERSION = 1

    async def async_step_user(self, user_input: dict[str, Any] | None = None
                              ) -> ConfigFlowResult:
        """Choose what to set up: conversation agent or speech-to-text."""
        return self.async_show_menu(step_id="user",
                                    menu_options=[TYPE_CONVERSATION, TYPE_STT])

    async def async_step_conversation(self, user_input: dict[str, Any] | None = None
                                      ) -> ConfigFlowResult:
        return await self._async_step_server(TYPE_CONVERSATION, "Local LLM",
                                             user_input)

    async def async_step_stt(self, user_input: dict[str, Any] | None = None
                             ) -> ConfigFlowResult:
        return await self._async_step_server(TYPE_STT, "Whisper", user_input)

    async def _async_step_server(self, kind: str, title: str,
                                 user_input: dict[str, Any] | None
                                 ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            url = user_input[CONF_URL].rstrip("/")
            await self.async_set_unique_id(url)
            self._abort_if_unique_id_configured()
            if (err := await _probe(self.hass, url, kind)) is not None:
                errors["base"] = err
            else:
                return self.async_create_entry(
                    title=title, data={CONF_URL: url, CONF_TYPE: kind})
        schema = vol.Schema({
            vol.Required(CONF_URL, default=_default_url(self.hass, kind)):
                TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
        })
        return self.async_show_form(step_id=kind, data_schema=schema,
                                    errors=errors)

    async def async_step_reconfigure(self, user_input: dict[str, Any] | None = None
                                     ) -> ConfigFlowResult:
        """Point an existing entry at another llama-server (add-on, GPU host)."""
        entry = self._get_reconfigure_entry()
        kind = entry.data.get(CONF_TYPE, TYPE_CONVERSATION)
        errors: dict[str, str] = {}
        if user_input is not None:
            url = user_input[CONF_URL].rstrip("/")
            if any(e.unique_id == url and e.entry_id != entry.entry_id
                   for e in self._async_current_entries(include_ignore=False)):
                return self.async_abort(reason="already_configured")
            if (err := await _probe(self.hass, url, kind)) is not None:
                errors["base"] = err
            else:
                # the entry's update listener reloads it
                return self.async_update_and_abort(
                    entry, unique_id=url, data_updates={CONF_URL: url},
                    reason="reconfigure_successful")
        schema = vol.Schema({
            vol.Required(CONF_URL, default=(user_input or entry.data)[CONF_URL]):
                TextSelector(TextSelectorConfig(type=TextSelectorType.URL)),
        })
        return self.async_show_form(step_id="reconfigure", data_schema=schema,
                                    errors=errors)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        return HaLlmAgentOptionsFlow()

    @classmethod
    @callback
    def async_supports_options_flow(cls, config_entry: ConfigEntry) -> bool:
        """Only the conversation agent has options."""
        return config_entry.data.get(CONF_TYPE, TYPE_CONVERSATION) == TYPE_CONVERSATION


class HaLlmAgentOptionsFlow(OptionsFlow):

    async def async_step_init(self, user_input: dict[str, Any] | None = None
                              ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        opts = self.config_entry.options
        schema = vol.Schema({
            vol.Optional(CONF_TEMPERATURE,
                         default=opts.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE)):
                NumberSelector(NumberSelectorConfig(
                    min=0, max=1, step=0.05, mode=NumberSelectorMode.SLIDER)),
            vol.Optional(CONF_MAX_HISTORY,
                         default=opts.get(CONF_MAX_HISTORY, DEFAULT_MAX_HISTORY)):
                NumberSelector(NumberSelectorConfig(
                    min=0, max=20, step=1, mode=NumberSelectorMode.BOX)),
            vol.Optional(CONF_TIMEOUT,
                         default=opts.get(CONF_TIMEOUT, DEFAULT_TIMEOUT)):
                NumberSelector(NumberSelectorConfig(
                    min=5, max=600, step=1, mode=NumberSelectorMode.BOX,
                    unit_of_measurement="s")),
            vol.Optional(CONF_MAX_ENTITIES,
                         default=opts.get(CONF_MAX_ENTITIES, DEFAULT_MAX_ENTITIES)):
                NumberSelector(NumberSelectorConfig(
                    min=10, max=1000, step=1, mode=NumberSelectorMode.BOX)),
            vol.Optional(CONF_USE_LOCATION,
                         default=opts.get(CONF_USE_LOCATION, DEFAULT_USE_LOCATION)):
                BooleanSelector(),
        })
        return self.async_show_form(step_id="init", data_schema=schema)
