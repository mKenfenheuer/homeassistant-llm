"""Consistency check for generated data: every assistant answer must agree
with the tool results before it.
Usage: python src/check_dataset.py [--lang de] FILE"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from collections import Counter

# per language (src/gen/<lang>/check.py, WORDS): words that report a failure,
# the word for "unreachable", and the sentence that lists switched-on devices
sys.path.insert(0, str(Path(__file__).resolve().parent))


def words(lang):
    return importlib.import_module(f"gen.{lang}.check").WORDS


W = words("de")
FAIL_WORDS = W["fail"]


def check(row):
    msgs = row["messages"]
    names = dict(re.findall(r"^(\S+\.\S+) \| (.+?) \| ", msgs[0]["content"], re.M))
    problems = []
    last_tool = None
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            res = json.loads(m["content"])
            # several get_state calls in one turn (two questions in one
            # sentence): the answer may use all of them
            if isinstance(res, list) and isinstance(last_tool, list):
                res = last_tool + res
            last_tool = res
            continue
        if m["role"] == "user":
            last_tool = None
            continue
        if m["role"] != "assistant" or m.get("tool_calls"):
            continue
        ans = m["content"] or ""
        if isinstance(last_tool, dict):           # control result
            # a failure is either reported or answered with a question back
            if last_tool.get("success") is False and not ans.rstrip().endswith("?") \
                    and not any(w in ans.lower() for w in FAIL_WORDS):
                problems.append(("error not reported", ans))
            if last_tool.get("success") is True and W["unreachable"] in ans:
                problems.append(("failure claimed after success", ans))
        elif isinstance(last_tool, list):         # get_state result
            blob = json.dumps(last_tool, ensure_ascii=False)
            for num in re.findall(r"\d+(?:[.,]\d+)?", ans):
                if num.replace(",", ".") not in blob and num not in blob and \
                        str(round(float(num.replace(",", ".")) * 100)) not in blob:
                    problems.append(("number not in tool result", ans))
                    break
            mo = re.match(W["listing"], ans)
            if mo:
                # only the listing sentence; a second answer can follow
                listed = re.split(W["list_split"], mo.group(1).split(". ")[0].rstrip("."))
                on = {x["name"] for x in last_tool if x.get("state") in ("on", "open")}
                if set(listed) - on:
                    problems.append(("listed entity not on/open", ans))
            if any(x.get("state") == "unavailable" for x in last_tool) and \
                    W["unreachable"] not in ans:
                problems.append(("unavailable not reported", ans))
        else:                                     # answer without tool in this turn
            pass
    return problems


def main():
    global W, FAIL_WORDS
    args = sys.argv[1:]
    if args[0] == "--lang":
        W = words(args[1])
        FAIL_WORDS = W["fail"]
        args = args[2:]
    rows = [json.loads(l) for l in open(args[0], encoding="utf-8")]
    c = Counter()
    examples = {}
    for r in rows:
        for kind, ans in check(r):
            c[kind] += 1
            examples.setdefault(kind, []).append((r["meta"]["cat"], ans))
    print(f"{len(rows)} rows, problems: {dict(c) or 'none'}")
    for k, ex in examples.items():
        for cat, a in ex[:4]:
            print(f"  [{k}] ({cat}) {a[:150]}")


if __name__ == "__main__":
    main()
