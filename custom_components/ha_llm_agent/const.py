"""Constants for the HA LLM Agent integration."""

DOMAIN = "ha_llm_agent"

CONF_URL = "url"
CONF_TEMPERATURE = "temperature"
CONF_MAX_HISTORY = "max_history"
CONF_TIMEOUT = "timeout"
CONF_MAX_ENTITIES = "max_entities"
CONF_USE_LOCATION = "use_location"

# hostname of the local add-on "ha_llm_inference" inside the HA network
DEFAULT_URL = "http://local-ha-llm-inference:8080"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_HISTORY = 4        # previous user turns kept per conversation
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
