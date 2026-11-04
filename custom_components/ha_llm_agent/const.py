"""Constants for the HA LLM Agent integration."""

DOMAIN = "ha_llm_agent"

CONF_URL = "url"
CONF_TEMPERATURE = "temperature"
CONF_MAX_HISTORY = "max_history"
CONF_TIMEOUT = "timeout"
CONF_MAX_ENTITIES = "max_entities"

# hostname of the local add-on "ha_llm_inference" inside the HA network
DEFAULT_URL = "http://local-ha-llm-inference:8080"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_HISTORY = 4        # previous user turns kept per conversation
DEFAULT_TIMEOUT = 120          # seconds per LLM request (CPU inference is slow)
DEFAULT_MAX_ENTITIES = 200

MAX_TOOL_ROUNDS = 4
MAX_CONVERSATIONS = 50
