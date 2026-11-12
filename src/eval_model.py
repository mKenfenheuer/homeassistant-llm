"""Evaluate a llama-server endpoint on the held-out set (data/eval.jsonl).

For each sample the conversation is cut before the assistant turn under test
and sent with the shared tool schema. Checks:
  - control: same (service, entity_id, data) triples as the gold call; an
    area target counts as the entity_ids it resolves to in the device list
  - get_state: all gold entity_ids requested (area resolved the same way)
  - unknown/chat: no tool call
  - question: after feeding the gold tool result, the answer contains the
    expected values

  - timer: same action, duration and name; get_time; broadcast with a message

Usage: python src/eval_model.py http://127.0.0.1:8080 [data/de/eval.jsonl] [N] [--lang de]
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

ARGS = sys.argv[1:]
LANG = "de"
if "--lang" in ARGS:
    i = ARGS.index("--lang")
    LANG = ARGS[i + 1]
    del ARGS[i:i + 2]
URL = ARGS[0].rstrip("/") if ARGS else "http://127.0.0.1:8080"
DATA = ARGS[1] if len(ARGS) > 1 else f"data/{LANG}/eval.jsonl"
LIMIT = int(ARGS[2]) if len(ARGS) > 2 else 10**9
TOOLS = S.build_tools(LANG)


def chat(messages):
    body = json.dumps({"model": "ha-llm", "messages": messages, "tools": TOOLS,
                       "temperature": 0, "max_tokens": 400}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())["choices"][0]["message"]


def norm_value(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(float(v), 2)
    if isinstance(v, str):
        return v.strip().casefold()
    return v


def timer_key(args):
    """Same action, same total duration, same timer name; the delayed
    command must be there when the gold has one (its wording is free)."""
    secs = sum(int(args.get(k) or 0) * f for k, f in (("hours", 3600), ("minutes", 60),
                                                      ("seconds", 1)))
    return (args.get("action"), secs, norm_value(args.get("name") or ""),
            bool(args.get("command")))


def devices(prompt):
    """(entity_id, name, area) tuples of the device list in the system prompt."""
    out = []
    for line in prompt[0]["content"].splitlines():
        p = [x.strip() for x in line.split(" | ")]
        if len(p) == 3 and "." in p[0] and " " not in p[0]:
            out.append((p[0], p[1], None if p[2] == "-" else p[2]))
    return out


def resolve(ids, area, domain, ents, groups):
    if ids:
        return [ids] if isinstance(ids, str) else ids
    if area and domain in S.AREA_DOMAINS:
        return S.area_targets(domain, area, ents, groups) or []
    return []


def triples(actions, ents=(), groups=()):
    out = set()
    for a in actions or []:
        dom = (a.get("service") or "").split(".", 1)[0]
        ids = resolve(a.get("entity_id"), a.get("area"), dom, ents, groups)
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


def check_call(gold, name, args, ents=(), groups=()):
    if gold is None:
        return name is None
    gname, gargs = gold
    if name != gname:
        return False
    if gname == "control":
        return triples(args.get("actions"), ents, groups) == \
            triples(gargs["actions"], ents, groups)
    if gname == "timer":
        return timer_key(args) == timer_key(gargs)
    if gname == "get_time":
        return True
    if gname == "broadcast":
        return bool(str(args.get("message") or "").strip())
    got = resolve(args.get("entity_ids"), args.get("area"),
                  args.get("domain") or "light", ents, groups)
    want = resolve(gargs.get("entity_ids"), gargs.get("area"),
                   gargs.get("domain") or "light", ents, groups)
    return set(want) <= set(got)


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
        ok = check_call(gold, name, args, devices(prompt), meta.get("groups", []))
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
