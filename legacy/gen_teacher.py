"""Teacher-generated part of the German SFT dataset, via an OpenAI-compatible
API (sub2api). The teacher only produces *German phrasings*; gold tool calls
are computed locally, so the training labels never depend on the teacher.

Two passes:
  1. paraphrase:  N natural German variants per command intent
  2. compound:    teacher also writes the action plan for multi-entity
                  commands; plans are validated against the schema, invalid
                  ones dropped.

Usage: python src/gen_teacher.py [out.jsonl]
Set TEACHER_API_KEY env var or pass --key.
"""

import json
import os
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import gen_template
from gen_template import (make_control_seed, render, ENTITY_INFO, NAME,
                          DOMAIN, COLORS, SYSTEM_PROMPT)
import ha_config
from ha_config import ALL_ACTION_NAMES, ACTIONS, tool_result

API = "https://sub2api.ksol.it/v1/chat/completions"
MODEL = "gpt-5.6-terra"
KEY = os.environ.get("TEACHER_API_KEY", "")


def chat(prompt, temperature=0.9):
    body = json.dumps({
        "model": MODEL,
        "temperature": temperature,
        "max_tokens": 1200,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(
        API, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + KEY})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(json.loads(r.read())["choices"][0]["message"]["content"])
        except Exception as e:
            if attempt == 3:
                raise
            import time
            time.sleep(5 * (attempt + 1))


def extract_json(txt):
    if isinstance(txt, (list, dict)):
        return txt
    try:
        return json.loads(txt)
    except Exception:
        import re
        m = re.search(r"\[.*\]|\{.*\}", txt, re.S)
        return json.loads(m.group(0)) if m else None


def validate_plan(plan):
    """Keep only fully-valid action plans."""
    if not isinstance(plan, dict) or "user" not in plan:
        return False
    acts = plan.get("actions")
    if not isinstance(acts, list) or not 1 <= len(acts) <= 5:
        return False
    for a in acts:
        eid, act = a.get("entity_id"), a.get("action")
        if eid not in ENTITY_INFO or act not in ACTIONS.get(DOMAIN.get(eid, ""), {}):
            return False
        data = a.get("data", {}) or {}
        spec = ACTIONS[DOMAIN[eid]][act]
        if set(data) != set(spec):
            return False
        for k, spec_v in spec.items():
            if isinstance(spec_v, tuple):
                lo, hi = spec_v
                if not isinstance(data[k], (int, float)) or not lo <= data[k] <= hi:
                    return False
    return True


# ---- pass 1: paraphrases -----------------------------------------------------
def paraphrase_job(seed):
    prompt = (
        "You collect German smart-home commands for a voice assistant dataset.\n"
        "Command intent: {intent} (target entities: {names}).\n"
        "Base phrasing: {base}\n"
        "Give 6 DIFFERENT natural German phrasings of the SAME command. "
        "Vary style: short imperative, polite, casual, start with 'hey', "
        "'please', 'so', 'quick', 'can you'. Keep the entity names exactly as "
        "written. Output ONLY a JSON array of 6 strings."
    ).format(intent=seed.get("_intent", seed["user"]),
             names=", ".join(NAME[a["entity_id"]] for a in seed["actions"]),
             base=seed["user"])
    variants = extract_json(chat(prompt))
    out = []
    if isinstance(variants, list):
        for v in variants[:6]:
            if isinstance(v, str) and v.strip():
                s = dict(seed)
                s["user"] = v
                out.append(s)
    return out


# ---- pass 2: compound commands ----------------------------------------------
def compound_job(_):
    table = "\n".join(
        "{eid} = {name} ({domain})".format(eid=e, name=info["name"], domain=info["domain"])
        for e, info in ENTITY_INFO.items())
    actions_txt = json.dumps(
        {d: {a: list(spec.keys()) for a, spec in dom.items()}
         for d, dom in ACTIONS.items() if dom}, ensure_ascii=False)
    prompt = (
        "You generate training data for a German smart-home assistant.\n"
        "Entities:\n{table}\n\n"
        "Allowed actions (domain: action with data fields):\n{actions_txt}\n\n"
        "Make 4 DIFFERENT realistic German user commands, each needing 2-4 "
        "actions at once (mix rooms; e.g. lights + shutters, alarm + lights, "
        "brightness for one lamp plus turning others off). "
        "Then give the action plan.\n"
        "Output ONLY a JSON array of 4 objects: "
        '[{{"user": "…german command…", "actions": '
        '[{{"action": "…", "entity_id": "…", "data": {{}}}}]}}]'
    ).format(table=table, actions_txt=actions_txt)
    plans = extract_json(chat(prompt, temperature=1.0))
    out = []
    if isinstance(plans, list):
        for p in plans:
            if validate_plan(p):
                out.append(p)
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/teacher.jsonl"
    random.Random(11)

    # pass 1 seeds: reuse the template seed generators
    seeds = []
    for eid, info in ENTITY_INFO.items():
        for action in ACTIONS[DOMAIN[eid]]:
            data = None
            if action == "light.set_brightness":
                data = {"brightness_pct": random.choice([10, 30, 50, 70, 90])}
            elif action == "light.set_color":
                data = {"color_hex": random.choice(list(COLORS.values()))}
            elif action == "thermostat.set_temperature":
                data = {"temperature": random.choice([17, 19, 21, 23])}
            s = make_control_seed(action, eid, data)
            s["_intent"] = action + " " + NAME[eid]
            seeds.append(s)
    jobs = [(paraphrase_job, (s,)) for s in seeds]
    for _ in range(10):
        jobs.append((compound_job, (0,)))

    results = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(fn, *args) for fn, args in jobs]
        for f in futs:
            results.append(f.result())

    rows = []
    sysmsg = {"role": "system", "content": gen_template.SYSTEM_PROMPT}
    for group in results:
        for seed in group or []:
            msgs = render(seed)
            rows.append({"messages": [sysmsg] + msgs})

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "rows to", path)


if __name__ == "__main__":
    main()
