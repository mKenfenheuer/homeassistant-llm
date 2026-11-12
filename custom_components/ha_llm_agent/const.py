"""Constants for the HA LLM Agent integration."""

DOMAIN = "ha_llm_agent"

CONF_URL = "url"
# kind of config entry: conversation agent (llama.cpp) or speech-to-text (whisper.cpp)
CONF_TYPE = "type"
TYPE_CONVERSATION = "conversation"
TYPE_STT = "stt"
TYPE_TTS = "tts"
CONF_TEMPERATURE = "temperature"
CONF_MAX_HISTORY = "max_history"
CONF_TIMEOUT = "timeout"
CONF_MAX_ENTITIES = "max_entities"
CONF_USE_LOCATION = "use_location"
# language of the model (llm_schema.LANGUAGES); empty: Home Assistant language
CONF_LANGUAGE = "language"
# answer length (llm_schema.VERBOSITY), written into the system prompt
CONF_VERBOSITY = "verbosity"

# hostname of the local add-on "ha_llm_inference" inside the HA network
DEFAULT_URL = "http://local-ha-llm-inference:8080"
# slug endings of the inference apps, preferred first. The NVIDIA and AMD/Intel
# apps of versions up to 0.9 are still found.
ADDON_SUFFIXES = ("ha_llm_inference", "ha_llm_inference_cuda", "ha_llm_inference_vulkan")
# hostname and slug ending of the speech-to-text app "ha_whisper"
DEFAULT_WHISPER_URL = "http://local-ha-whisper:8080"
WHISPER_ADDON_SUFFIXES = ("ha_whisper",)
STT_TIMEOUT = 60               # seconds per transcription
# hostname and slug ending of the text-to-speech app "ha_tts"
DEFAULT_TTS_URL = "http://local-ha-tts:8080"
TTS_ADDON_SUFFIXES = ("ha_tts",)
TTS_TIMEOUT = 120              # seconds per sentence
# reference recordings for voice cloning (Qwen3-TTS), shared with the app
TTS_VOICE_DIR = "/share/ha_tts/voices"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_HISTORY = 6        # previous user turns kept per conversation
DEFAULT_TIMEOUT = 300          # seconds per LLM request (CPU inference is slow)
DEFAULT_MAX_ENTITIES = 200
# The area line sits before the tool block in the prompt; a different area per
# satellite invalidates ~600 cached tokens. Off by default for slow CPUs.
DEFAULT_USE_LOCATION = False

MAX_TOOL_ROUNDS = 4
MAX_CONVERSATIONS = 50
WARMUP_TIMEOUT = 1800          # first prefill on a slow CPU can take minutes
WARMUP_INTERVAL_MIN = 5        # re-check the device list every N minutes
SLOW_WARMUP_S = 60             # above: repair issue "slow inference"
