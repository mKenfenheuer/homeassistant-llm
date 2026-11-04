"""Evaluate a llama-server endpoint on the held-out set (data/eval.jsonl).

For each sample the conversation is cut before the assistant turn under test
and sent with the shared tool schema. Checks:
  - control: same (service, entity_id, data) triples as the gold call
  - get_state: all gold entity_ids requested
  - unknown/chat: no tool call
  - question: after feeding the gold tool result, the answer contains the
    expected values

Usage: python src/eval_model.py http://127.0.0.1:8080 [data/eval.jsonl] [N]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "custom_components" / "ha_llm_agent"))
import llm_schema as S  # noqa: E402

URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8080"
DATA = sys.argv[2] if len(sys.argv) > 2 else "data/eval.jsonl"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9


def chat(messages):
    body = json.dumps({"model": "ha-llm", "messages": messages, "tools": S.TOOLS,
                       "temperature": 0, "max_tokens": 400}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]


def norm_value(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 2)
    return v


def triples(actions):
    out = set()
    for a in actions or []:
        ids = a.get("entity_id") or []
        if isinstance(ids, str):
            ids = [ids]
        data = tuple(sorted((k, norm_value(v)) for k, v in (a.get("data") or {}).items()))
        for e in ids:
            out.add((a.get("service"), e, data))
    return out


def first_call(msg):
    calls = msg.get("tool_calls") or []
    if not calls:
        return None, None
    fn = calls[0]["function"]
    try:
        args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
    except ValueError:
        args = {"_invalid": fn["arguments"]}
    return fn["name"], args


def check_call(gold, name, args):
    if gold is None:
        return name is None
    gname, gargs = gold
    if name != gname:
        return False
    if gname == "control":
        return triples(args.get("actions")) == triples(gargs["actions"])
    return set(gargs["entity_ids"]) <= set(args.get("entity_ids") or [])


def main():
    rows = [json.loads(l) for l in open(DATA, encoding="utf-8")][:LIMIT]
    stats = defaultdict(lambda: [0, 0])
    answer_stats = [0, 0]
    fails = []
    t0 = time.time()
    for row in rows:
        msgs, meta = row["messages"], row["meta"]
        cat = meta["cat"]
        # cut before the assistant turn under test
        last_user = max(i for i, m in enumerate(msgs) if m["role"] == "user")
        prompt = msgs[:last_user + 1]
        if "gold_last" in meta:
            gold = meta["gold_last"][0] if meta["gold_last"] else None
        else:
            gold = meta["gold"][0] if meta["gold"] else None
        msg = chat(prompt)
        name, args = first_call(msg)
        ok = check_call(gold, name, args)
        stats[cat][0] += ok
        stats[cat][1] += 1
        if not ok:
            fails.append((cat, prompt[-1]["content"], gold, name, args,
                          (msg.get("content") or "")[:120]))
        # grounded answer check for questions
        if cat == "question" and ok:
            answer = chat(msgs[:last_user + 3])["content"] or ""
            hit = all(s.lower() in answer.lower() for s in meta["answer_contains"])
            answer_stats[0] += hit
            answer_stats[1] += 1
            if not hit:
                fails.append(("answer", msgs[last_user]["content"], meta["answer_contains"],
                              None, None, answer[:160]))
    dt = time.time() - t0
    total = sum(v[0] for v in stats.values()), sum(v[1] for v in stats.values())
    print(f"\n== {URL}  {len(rows)} samples, {dt:.0f}s ({dt / len(rows):.2f}s/sample)")
    for cat, (ok, n) in sorted(stats.items()):
        print(f"  {cat:10s} {ok:4d}/{n:<4d} {100 * ok / n:5.1f}%")
    print(f"  {'TOOL CALLS':10s} {total[0]:4d}/{total[1]:<4d} {100 * total[0] / total[1]:5.1f}%")
    if answer_stats[1]:
        print(f"  {'ANSWERS':10s} {answer_stats[0]:4d}/{answer_stats[1]:<4d} "
              f"{100 * answer_stats[0] / answer_stats[1]:5.1f}%")
    print("\nfirst failures:")
    for f in fails[:15]:
        print("  ", json.dumps(f, ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
