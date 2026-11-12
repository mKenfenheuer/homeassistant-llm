"""German dataset generator: device intents (core) plus Assist built-ins."""

from . import assist  # noqa: F401  registers its hooks, intents and cases
from .core import main  # noqa: F401
