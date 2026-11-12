"""Generate the SFT dataset for one language.

Every language has its own generator package src/gen/<lang>/ with a
main(argv) function, because sentences, grammar and device names differ per
language. The tool schema and the prompt come from
custom_components/ha_llm_agent/llm_schema.py (LANGUAGES[<lang>]).

Usage: python src/gen_dataset.py [--lang de] OUT.jsonl N|per_case:K [SEED]
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    argv = sys.argv[1:]
    lang = "de"
    if argv and argv[0] == "--lang":
        lang, argv = argv[1], argv[2:]
    try:
        gen = importlib.import_module(f"gen.{lang}")
    except ModuleNotFoundError:
        sys.exit(f"no dataset generator for language {lang!r} (src/gen/{lang}/)")
    gen.main(argv)


if __name__ == "__main__":
    main()
