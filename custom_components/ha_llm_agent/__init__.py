"""HA LLM Agent: local conversation agent (llama.cpp) and speech-to-text (whisper.cpp)."""

from __future__ import annotations

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .const import CONF_TYPE, TYPE_CONVERSATION, TYPE_STT, TYPE_TTS

# entries of earlier versions have no type and are conversation agents
PLATFORMS = {
    TYPE_CONVERSATION: [Platform.CONVERSATION],
    TYPE_STT: [Platform.STT],
    TYPE_TTS: [Platform.TTS],
}


def _platforms(entry: ConfigEntry) -> list[Platform]:
    return PLATFORMS[entry.data.get(CONF_TYPE, TYPE_CONVERSATION)]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    await hass.config_entries.async_forward_entry_setups(entry, _platforms(entry))
    entry.async_on_unload(entry.add_update_listener(_async_reload))
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    return await hass.config_entries.async_unload_platforms(entry, _platforms(entry))


async def _async_reload(hass: HomeAssistant, entry: ConfigEntry) -> None:
    await hass.config_entries.async_reload(entry.entry_id)
