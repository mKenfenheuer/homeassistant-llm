"""Consistency check for generated data: every assistant answer must agree
with the tool results before it. Usage: python src/check_dataset.py FILE"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

FAIL_WORDS = ("nicht", "fehler", "erreichbar", "teilweise", "leider")


def check(row):
    msgs = row["messages"]
    names = dict(re.findall(r"^(\S+\.\S+) \| (.+?) \| ", msgs[0]["content"], re.M))
    problems = []
    last_tool = None
    for i, m in enumerate(msgs):
        if m["role"] == "tool":
            last_tool = json.loads(m["content"])
            continue
        if m["role"] == "user":
            last_tool = None
            continue
        if m["role"] != "assistant" or m.get("tool_calls"):
            continue
        ans = m["content"] or ""
        if isinstance(last_tool, dict):           # control result
            if last_tool.get("success") is False and not any(w in ans.lower() for w in FAIL_WORDS):
                problems.append(("error not reported", ans))
            if last_tool.get("success") is True and "erreichbar" in ans:
                problems.append(("failure claimed after success", ans))
        elif isinstance(last_tool, list):         # get_state result
            blob = json.dumps(last_tool, ensure_ascii=False)
            for num in re.findall(r"\d+(?:,\d+)?", ans):
                if num.replace(",", ".") not in blob and num not in blob and \
                        str(round(float(num.replace(",", ".")) * 100)) not in blob:
                    problems.append(("number not in tool result", ans))
                    break
            mo = re.match(r"(?:Ja\. |Nein\. )?(?:Offen|An) sind?: (.+)\.$", ans)
            if mo:
                listed = re.split(r", | und ", mo.group(1))
                on = {x["name"] for x in last_tool if x.get("state") == "on"}
                if set(listed) - on:
                    problems.append(("listed entity not on/open", ans))
            if any(x.get("state") == "unavailable" for x in last_tool) and \
                    "erreichbar" not in ans:
                problems.append(("unavailable not reported", ans))
        else:                                     # answer without tool in this turn
            pass
    return problems


def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
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
