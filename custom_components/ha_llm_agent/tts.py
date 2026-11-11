"""Text-to-speech with the local llama-server of the app "HA TTS" (POST /tts)."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncGenerator
import io
import logging
from pathlib import Path
import re
import struct
from typing import Any
import wave

import aiohttp

from homeassistant.components import tts
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from homeassistant.helpers.device_registry import DeviceEntryType, DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from .const import CONF_URL, DOMAIN, TTS_TIMEOUT, TTS_VOICE_DIR
from .spoken import numbers_to_words

_LOGGER = logging.getLogger(__name__)

# Languages per model, as llama.cpp documents them for --tts-lang
MODEL_LANGUAGES = {
    "lfm3-omni": ["ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "it",
                  "ja", "ko", "nl", "pl", "pt", "sv", "tr", "zh"],
    "qwen3-tts": ["de", "en", "es", "fr", "it", "ja", "ko", "pt", "ru", "zh"],
}
# Qwen3-TTS without reference recording
DEFAULT_VOICE = "default"
VOICE_SUFFIXES = (".wav", ".mp3")
# A sentence ends with . ! ? or a line break, followed by whitespace
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|\n+")
WAV_RATE = 24000

# Units after a number, written out per language. TTS models read symbols
# like "°C" as letters, Home Assistant states use them all the time.
UNITS = {
    "de": {"°C": "Grad Celsius", "°F": "Grad Fahrenheit", "°": "Grad", "%": "Prozent",
           "kWh": "Kilowattstunden", "Wh": "Wattstunden", "kW": "Kilowatt", "W": "Watt",
           "V": "Volt", "mA": "Milliampere", "A": "Ampere", "km/h": "Kilometer pro Stunde",
           "m/s": "Meter pro Sekunde", "hPa": "Hektopascal", "lx": "Lux", "ppm": "ppm",
           "µg/m³": "Mikrogramm pro Kubikmeter", "m³": "Kubikmeter", "mm": "Millimeter",
           "cm": "Zentimeter", "km": "Kilometer", "min": "Minuten", "h": "Stunden",
           "kg": "Kilogramm", "l": "Liter", "dB": "Dezibel", "€": "Euro"},
    "en": {"°C": "degrees Celsius", "°F": "degrees Fahrenheit", "°": "degrees",
           "%": "percent", "kWh": "kilowatt hours", "Wh": "watt hours", "kW": "kilowatts",
           "W": "watts", "V": "volts", "mA": "milliamps", "A": "amps",
           "km/h": "kilometers per hour", "m/s": "meters per second",
           "hPa": "hectopascals", "lx": "lux", "ppm": "ppm",
           "µg/m³": "micrograms per cubic meter", "m³": "cubic meters",
           "mm": "millimeters", "cm": "centimeters", "km": "kilometers",
           "min": "minutes", "h": "hours", "kg": "kilograms", "l": "liters",
           "dB": "decibels", "€": "euros"},
}


def normalize_text(text: str, language: str) -> str:
    """Write units after numbers, then the numbers themselves as words."""
    units = UNITS.get(language.split("-")[0].lower())
    if units:
        pattern = "|".join(re.escape(u) for u in sorted(units, key=len, reverse=True))
        text = re.sub(rf"(\d)\s?({pattern})(?![\w/-])",
                      lambda m: f"{m.group(1)} {units[m.group(2)]}", text)
    return numbers_to_words(text, language)


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry,
                            async_add_entities: AddEntitiesCallback) -> None:
    url = entry.data[CONF_URL].rstrip("/")
    session = async_get_clientsession(hass)
    try:
        async with session.get(f"{url}/props",
                               timeout=aiohttp.ClientTimeout(total=10)) as resp:
            resp.raise_for_status()
            props = await resp.json(content_type=None)
    except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as err:
        raise ConfigEntryNotReady(f"TTS server at {url} not ready: {err}") from err

    model = props.get("model_alias") or "lfm3-omni"
    server_voices = props.get("voices") or []
    voice_files = ([] if server_voices
                   else await hass.async_add_executor_job(_list_voice_files))
    async_add_entities([HaTtsEntity(entry, url, model, server_voices, voice_files)])


def _list_voice_files() -> list[str]:
    path = Path(TTS_VOICE_DIR)
    if not path.is_dir():
        return []
    return sorted(f.name for f in path.iterdir()
                  if f.is_file() and f.suffix.lower() in VOICE_SUFFIXES)


def _wav_header(rate: int) -> bytes:
    """WAV header with 0 frames: the length isn't known while streaming."""
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(rate)
        return buf.getvalue()


def _float_to_pcm16(data: bytes) -> bytes:
    count = len(data) // 4
    samples = struct.unpack(f"<{count}f", data[: count * 4])
    return struct.pack(f"<{count}h", *(
        max(-32768, min(32767, int(s * 32767))) for s in samples))


class HaTtsEntity(tts.TextToSpeechEntity):
    """Speaks sentence by sentence while the text is still arriving."""

    # TTS needs an engine name, an entity named after its device has none
    _attr_supported_options = [tts.ATTR_VOICE]

    def __init__(self, entry: ConfigEntry, url: str, model: str,
                 server_voices: list[str], voice_files: list[str]) -> None:
        self._url = url
        self._model = model
        self._server_voices = server_voices
        self._voice_files = voice_files
        self._attr_unique_id = entry.entry_id
        self._attr_name = entry.title
        self._attr_supported_languages = MODEL_LANGUAGES.get(
            model, MODEL_LANGUAGES["lfm3-omni"])
        self._attr_default_language = (
            "de" if "de" in self._attr_supported_languages
            else self._attr_supported_languages[0])
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="llama.cpp",
            model=model,
            entry_type=DeviceEntryType.SERVICE,
        )

    async def async_load_voices(self) -> None:
        """Read the reference recordings, which the user can add at any time."""
        if not self._server_voices:
            self._voice_files = await self.hass.async_add_executor_job(
                _list_voice_files)

    def _voice_ids(self) -> list[str]:
        if self._server_voices:
            return list(self._server_voices)
        return [DEFAULT_VOICE, *self._voice_files]

    @property
    def default_options(self) -> dict[str, Any]:
        return {tts.ATTR_VOICE: self._voice_ids()[0]}

    @callback
    def async_get_supported_voices(self, language: str) -> list[tts.Voice] | None:
        return [tts.Voice(v, Path(v).stem if v != DEFAULT_VOICE else "Default")
                for v in self._voice_ids()]

    async def _voice_payload(self, voice: str | None) -> dict[str, Any]:
        if self._server_voices:
            if voice in self._server_voices:
                return {"voice": voice}
            return {}
        await self.async_load_voices()
        if not voice or voice == DEFAULT_VOICE or voice not in self._voice_files:
            return {}

        def read() -> bytes:
            return (Path(TTS_VOICE_DIR) / voice).read_bytes()

        data = await self.hass.async_add_executor_job(read)
        return {"speaker_ref_b64": base64.b64encode(data).decode()}

    async def _speak(self, text: str, language: str, voice_payload: dict[str, Any]
                     ) -> AsyncGenerator[tuple[int, bytes]]:
        """Stream one sentence as (sample rate, 16 bit PCM) chunks."""
        body = {"input": normalize_text(text, language),
                "lang": language.split("-")[0].lower(),
                "stream": True, "response_format": "pcm", **voice_payload}
        session = async_get_clientsession(self.hass)
        try:
            # llama-server closes the connection after a streamed answer, so
            # don't reuse it for the next sentence
            async with session.post(
                f"{self._url}/tts", json=body, headers={"Connection": "close"},
                timeout=aiohttp.ClientTimeout(total=TTS_TIMEOUT),
            ) as resp:
                resp.raise_for_status()
                match = re.search(r"rate=(\d+)", resp.headers.get("Content-Type", ""))
                rate = int(match.group(1)) if match else WAV_RATE
                rest = b""
                async for chunk in resp.content.iter_any():
                    data = rest + chunk
                    usable = len(data) - len(data) % 4
                    rest = data[usable:]
                    if usable:
                        yield rate, _float_to_pcm16(data[:usable])
        except (aiohttp.ClientError, asyncio.TimeoutError) as err:
            raise HomeAssistantError(f"TTS server at {self._url} failed: {err}") from err

    async def async_stream_tts_audio(
        self, request: tts.TTSAudioRequest
    ) -> tts.TTSAudioResponse:
        voice_payload = await self._voice_payload(request.options.get(tts.ATTR_VOICE))
        sentences: asyncio.Queue[str | None] = asyncio.Queue()

        async def split_text() -> None:
            # Hand over every complete sentence while the text still arrives
            buffer = ""
            try:
                async for chunk in request.message_gen:
                    buffer += chunk
                    parts = SENTENCE_END.split(buffer)
                    for sentence in parts[:-1]:
                        if sentence.strip():
                            await sentences.put(sentence.strip())
                    buffer = parts[-1]
                if buffer.strip():
                    await sentences.put(buffer.strip())
            finally:
                await sentences.put(None)

        async def data_gen() -> AsyncGenerator[bytes]:
            splitter = asyncio.create_task(split_text())
            header_sent = False
            try:
                while (sentence := await sentences.get()) is not None:
                    async for rate, pcm in self._speak(
                            sentence, request.language, voice_payload):
                        if not header_sent:
                            yield _wav_header(rate)
                            header_sent = True
                        yield pcm
            finally:
                splitter.cancel()

        return tts.TTSAudioResponse("wav", data_gen())

    async def async_get_tts_audio(
        self, message: str, language: str, options: dict[str, Any]
    ) -> tts.TtsAudioType:
        voice_payload = await self._voice_payload(options.get(tts.ATTR_VOICE))
        pcm = bytearray()
        rate = WAV_RATE
        for sentence in (s.strip() for s in SENTENCE_END.split(message)):
            if not sentence:
                continue
            async for rate, chunk in self._speak(sentence, language, voice_payload):
                pcm.extend(chunk)
        if not pcm:
            return None, None
        with io.BytesIO() as buf:
            with wave.open(buf, "wb") as out:
                out.setnchannels(1)
                out.setsampwidth(2)
                out.setframerate(rate)
                out.writeframes(bytes(pcm))
            return "wav", buf.getvalue()
