"""Speech-to-text with the local whisper.cpp server (app "HA Whisper")."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable
import io
import logging
import wave

import aiohttp

from homeassistant.components import stt
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import llm_schema as S
from .const import CONF_URL, DOMAIN, STT_TIMEOUT

_LOGGER = logging.getLogger(__name__)

# Languages of the multilingual Whisper models
WHISPER_LANGUAGES = [
    "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs",
    "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi",
    "fo", "fr", "gl", "gu", "ha", "haw", "he", "hi", "hr", "ht", "hu", "hy",
    "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb",
    "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt",
    "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru",
    "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw",
    "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "uk", "ur", "uz", "vi",
    "yi", "yo", "yue", "zh",
]


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry,
                            async_add_entities: AddEntitiesCallback) -> None:
    async_add_entities([HaWhisperSttEntity(entry)])


class HaWhisperSttEntity(stt.SpeechToTextEntity):
    """Sends the recorded speech to whisper-server and returns the text."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: ConfigEntry) -> None:
        self._url = entry.data[CONF_URL].rstrip("/")
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="whisper.cpp",
            model="whisper-server",
            entry_type=DeviceEntryType.SERVICE,
        )

    @property
    def supported_languages(self) -> list[str]:
        return WHISPER_LANGUAGES

    @property
    def supported_formats(self) -> list[stt.AudioFormats]:
        return [stt.AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[stt.AudioCodecs]:
        return [stt.AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[stt.AudioBitRates]:
        return [stt.AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[stt.AudioSampleRates]:
        return [stt.AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[stt.AudioChannels]:
        return [stt.AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: stt.SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> stt.SpeechResult:
        # whisper.cpp transcribes complete recordings, so collect the stream
        pcm = bytearray()
        async for chunk in stream:
            pcm.extend(chunk)
        if not pcm:
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        wav = io.BytesIO()
        with wave.open(wav, "wb") as out:
            out.setnchannels(metadata.channel.value)
            out.setsampwidth(metadata.bit_rate.value // 8)
            out.setframerate(metadata.sample_rate.value)
            out.writeframes(bytes(pcm))

        form = aiohttp.FormData()
        form.add_field("file", wav.getvalue(), filename="speech.wav",
                       content_type="audio/wav")
        # "de-DE" -> "de"
        lang = metadata.language.split("-")[0].lower()
        form.add_field("language", lang)
        # the command vocabulary keeps "Musik pausieren" from becoming
        # "Musik passieren"
        if lang in S.LANGUAGES:
            form.add_field("prompt", S.LANGUAGES[lang]["stt_prompt"])
        form.add_field("response_format", "json")
        form.add_field("temperature", "0.0")

        session = async_get_clientsession(self.hass)
        try:
            async with session.post(
                f"{self._url}/inference", data=form,
                timeout=aiohttp.ClientTimeout(total=STT_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                result = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
            _LOGGER.warning("whisper-server at %s failed: %s", self._url, err)
            return stt.SpeechResult(None, stt.SpeechResultState.ERROR)

        text = (result.get("text") or "").strip()
        _LOGGER.debug("Recognized (%s, %.1f s audio): %s", metadata.language,
                      len(pcm) / (metadata.sample_rate.value * 2), text)
        return stt.SpeechResult(text, stt.SpeechResultState.SUCCESS)
