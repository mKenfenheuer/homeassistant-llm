"""Check that all languages stay in step.

1. llm_schema.LANGUAGES: every language has the same keys, the same
   placeholders in every text, 7 weekdays, 12 months, every weather condition,
   every answer length and every unit.
2. Generators (src/gen/<lang>/): same case names as German, and per case the
   same sample category and the same tools and services. A (tool, service)
   pair that one language uses in at least MIN_SHARE of a case's samples and
   another language never uses in that case is a mismatch: one generator got
   a change the others did not. (Rarer pairs come from random history turns
   and differ between seeds even within one language.)
3. Every row has "locale" set to its language and a verbosity level.

Usage: python src/check_languages.py [--langs de,en,...] [--per-case 120]
Exit code 1 on a mismatch. German is the reference.
"""

from __future__ import annotations

import json
import re
import string
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

REF = "de"
MIN_SHARE = 0.15


def placeholders(text: str) -> set[str]:
    return {f for _, f, _, _ in string.Formatter().parse(text) if f}


def check_texts(langs) -> list[str]:
    errs = []
    ref = S.LANGUAGES[REF]
    for lang in langs:
        t = S.LANGUAGES.get(lang)
        if t is None:
            errs.append(f"{lang}: no entry in llm_schema.LANGUAGES")
            continue
        if set(t) != set(ref):
            errs.append(f"{lang}: keys differ from {REF}: {sorted(set(t) ^ set(ref))}")
        for k, v in ref.items():
            if isinstance(v, str) and isinstance(t.get(k), str) and \
                    placeholders(v) != placeholders(t[k]):
                errs.append(f"{lang}: placeholders of {k}: {placeholders(t[k])} != {placeholders(v)}")
        if len(t.get("weekdays", [])) != 7 or len(t.get("months", [])) != 12:
            errs.append(f"{lang}: needs 7 weekdays and 12 months")
        if set(t.get("conditions", {})) != set(S.WEATHER_CONDITIONS):
            errs.append(f"{lang}: weather conditions differ")
        if set(t.get("verbosity", {})) != set(S.VERBOSITY) or \
                set(t.get("verbosity_labels", {})) != set(S.VERBOSITY):
            errs.append(f"{lang}: answer lengths differ")
        if set(t.get("units", {})) != {"hours", "minutes", "seconds"}:
            errs.append(f"{lang}: units differ")
        if not (ROOT / f"templates/chat_template.{lang}.jinja").exists():
            errs.append(f"{lang}: templates/chat_template.{lang}.jinja missing")
    return errs


def signature(row):
    """(tool, service) pairs of every tool call in the sample."""
    out = set()
    for m in row["messages"]:
        for c in m.get("tool_calls") or []:
            fn = c["function"]
            args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
            if fn["name"] == "control":
                for a in args.get("actions", []):
                    out.add(("control", a.get("service")))
            elif fn["name"] == "timer":
                out.add(("timer", args.get("action")))
            else:
                out.add((fn["name"], ""))
    return out


def generate(lang, per_case, tmp) -> list[dict]:
    out = Path(tmp) / f"{lang}.jsonl"
    subprocess.run([sys.executable, str(ROOT / "src/gen_dataset.py"), "--lang", lang, str(out),
                    f"per_case:{per_case}", "7"], check=True, capture_output=True, text=True)
    return [json.loads(l) for l in open(out, encoding="utf-8")]


def profile(rows):
    cats, pairs = defaultdict(Counter), defaultdict(Counter)
    for r in rows:
        case = r["meta"]["case"]
        cats[case][r["meta"]["cat"]] += 1
        for p in signature(r):
            pairs[case][p] += 1
    return cats, pairs


def check_generators(langs, per_case) -> list[str]:
    errs = []
    with tempfile.TemporaryDirectory() as tmp:
        data = {}
        for lang in [REF] + [l for l in langs if l != REF]:
            if not (ROOT / "src/gen" / lang / "__init__.py").exists():
                errs.append(f"{lang}: no generator in src/gen/{lang}/")
                continue
            try:
                data[lang] = generate(lang, per_case, tmp)
            except subprocess.CalledProcessError as err:
                errs.append(f"{lang}: generator failed: {err.stderr.strip().splitlines()[-1:]}")
        if REF not in data:
            return errs + [f"reference {REF} could not be generated"]
        ref_cats, ref_pairs = profile(data[REF])
        for lang, rows in data.items():
            bad_rows = [r for r in rows if r.get("locale") != lang or "verbosity" not in r["meta"]]
            if bad_rows:
                errs.append(f"{lang}: {len(bad_rows)} rows without locale={lang} or verbosity")
            if lang == REF:
                continue
            cats, pairs = profile(rows)
            if set(cats) != set(ref_cats):
                errs.append(f"{lang}: cases differ: missing {sorted(set(ref_cats) - set(cats))}, "
                            f"extra {sorted(set(cats) - set(ref_cats))}")
            for case in sorted(set(cats) & set(ref_cats)):
                if set(cats[case]) != set(ref_cats[case]):
                    errs.append(f"{lang} {case}: categories {dict(cats[case])} != {REF} {dict(ref_cats[case])}")
                n_ref, n_lang = sum(ref_cats[case].values()), sum(cats[case].values())
                for p, n in ref_pairs[case].items():
                    if n >= MIN_SHARE * n_ref and pairs[case][p] == 0:
                        errs.append(f"{lang} {case}: never uses {p} ({REF}: {n}/{n_ref})")
                for p, n in pairs[case].items():
                    if n >= MIN_SHARE * n_lang and ref_pairs[case][p] == 0:
                        errs.append(f"{lang} {case}: uses {p} {n}/{n_lang}, {REF} never")
    return errs


def main():
    argv = sys.argv[1:]
    langs = list(S.LANGUAGES)
    per_case = 120
    if "--langs" in argv:
        langs = argv[argv.index("--langs") + 1].split(",")
    if "--per-case" in argv:
        per_case = int(argv[argv.index("--per-case") + 1])
    errs = check_texts(langs) + check_generators(langs, per_case)
    for e in errs:
        print("MISMATCH", e)
    print(f"{len(langs)} languages, {len(errs)} mismatches")
    sys.exit(1 if errs else 0)


if __name__ == "__main__":
    main()
