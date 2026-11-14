"""Every name in the acceptance house must be new to the language's training
data: no device or area name may appear in src/gen/<lang>/.

    python tests/acceptance/check_names.py en de
"""
import importlib
import sys
from pathlib import Path

GEN = Path(__file__).resolve().parents[2] / "src" / "gen"
bad = 0
for lang in sys.argv[1:]:
    L = importlib.import_module(lang)
    src = "\n".join(p.read_text(encoding="utf-8") for p in (GEN / lang).glob("*.py")).lower()
    for n in list(L.NAME.values()) + list(L.AREA.values()):
        if f'"{n.lower()}"' in src or f"'{n.lower()}'" in src or (" " in n and n.lower() in src):
            print(f"{lang}: known to the generator: {n}")
            bad += 1
print("ok" if not bad else f"{bad} known names")
sys.exit(1 if bad else 0)
