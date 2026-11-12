"""Generate the datasets of several languages and merge them.

For every language: src/gen_dataset.py writes data/<lang>/train.jsonl and
data/<lang>/eval.jsonl, src/check_dataset.py checks them. Then all rows are
merged into data/all/train.jsonl and data/all/eval.jsonl, each with its
"locale" column, and src/check_languages.py checks that the generators are in
step. src/upload_hf.py --all uploads the result.

Usage: python src/gen_all.py [--langs de,en,es,fr,it,nl] [--per-case 236]
                             [--seed 16] [--eval 300] [--no-parity]
Default: all languages that have a generator in src/gen/<lang>/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def available() -> list[str]:
    return sorted(p.name for p in (ROOT / "src/gen").iterdir()
                  if (p / "__init__.py").exists())


def opt(argv, name, default):
    return argv[argv.index(name) + 1] if name in argv else default


def run(*args) -> str:
    res = subprocess.run([PY, *map(str, args)], cwd=ROOT, check=True,
                         capture_output=True, text=True)
    return res.stdout.strip()


def merge(langs, split) -> Counter:
    out = ROOT / "data/all" / f"{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    with open(out, "w", encoding="utf-8") as fh:
        for lang in langs:
            for line in open(ROOT / "data" / lang / f"{split}.jsonl", encoding="utf-8"):
                row = json.loads(line)
                row = {"locale": lang, **{k: v for k, v in row.items() if k != "locale"}}
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                counts[lang] += 1
    return counts


def main():
    argv = sys.argv[1:]
    langs = opt(argv, "--langs", ",".join(available())).split(",")
    per_case = int(opt(argv, "--per-case", 236))
    seed = int(opt(argv, "--seed", 16))
    n_eval = int(opt(argv, "--eval", 300))
    missing = [l for l in langs if l not in available()]
    if missing:
        sys.exit(f"no generator for {missing} (src/gen/<lang>/)")
    for lang in langs:
        d = ROOT / "data" / lang
        print(f"[{lang}]", run("src/gen_dataset.py", "--lang", lang, d / "train.jsonl",
                               f"per_case:{per_case}", seed).splitlines()[-1])
        run("src/gen_dataset.py", "--lang", lang, d / "eval.jsonl", n_eval, seed + 1000)
        for split in ("train", "eval"):
            res = run("src/check_dataset.py", "--lang", lang, d / f"{split}.jsonl")
            print(f"[{lang}] {split}: {res.splitlines()[0]}")
            if "problems: none" not in res:
                sys.exit(f"[{lang}] {split}: consistency check failed\n{res}")
    for split in ("train", "eval"):
        print(f"data/all/{split}.jsonl:", dict(merge(langs, split)))
    if "--no-parity" not in argv and len(langs) > 1:
        res = subprocess.run([PY, "src/check_languages.py", "--langs", ",".join(langs)],
                             cwd=ROOT, capture_output=True, text=True)
        print(res.stdout.strip().splitlines()[-1])
        if res.returncode:
            sys.exit(res.stdout)


if __name__ == "__main__":
    main()
