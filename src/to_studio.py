"""Write a language's dataset in the form AI Studio trains on, and check that
AI Studio would learn exactly what train_sft.py learns from it.

AI Studio (github.com/mKenfenheuer/ai-studio) takes chat rows with the tool
schema on each row; train_sft.py adds the schema at load time from
llm_schema.build_tools. So each row gets `tools`, and the rows train_sft.py
drops for being longer than --max-len tokens are dropped here too (AI Studio
would cut them, losing the answer at the end). The held-out file is every
--eval-every-th remaining eval row, the same 100 rows train_sft.py measures.

--check N renders N rows both ways -- train_sft.encode, and AI Studio's own
code (common.conversation with the model's chat template, the same steps as
runner/jobs/lora_llm.py) -- and compares the token ids and the trained
tokens. It needs the AI Studio checkout (--studio) and transformers.

Usage:
    python src/to_studio.py --lang en --out data/en/studio [--check 300 --studio ../ai-studio]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "custom_components" / "ha_llm_agent"))
sys.path.insert(0, str(ROOT / "src"))
import llm_schema  # noqa: E402


def studio_encode(tok, row: dict, fmt: dict, conversation, max_len: int):
    """runner/jobs/lora_llm.py tokenize(), for one row."""
    conv, _ = conversation.repair(conversation.from_row(row, fmt))
    text, spans, exact = conversation.trainable_spans(conv, fmt, "assistant")
    if not text.rstrip().endswith(tok.eos_token):
        if spans and spans[-1][1] == len(text):
            spans[-1] = (spans[-1][0], len(text) + len(tok.eos_token))
        text += tok.eos_token
    enc = tok(text, truncation=True, max_length=max_len, padding=False,
              return_offsets_mapping=True)
    labels = []
    for tid, (start, end) in zip(enc["input_ids"], enc["offset_mapping"]):
        inside = any(s <= start and (end or start + 1) <= e for s, e in spans)
        labels.append(tid if inside else -100)
    return enc["input_ids"], labels, exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--studio", default=str(ROOT.parent / "ai-studio"))
    a = ap.parse_args()
    out = Path(a.out or ROOT / "data" / a.lang / "studio")
    out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    import train_sft
    tok = AutoTokenizer.from_pretrained(a.model)
    tools = llm_schema.build_tools(a.lang)

    def rows(name):
        for line in open(ROOT / "data" / a.lang / name, encoding="utf-8"):
            r = json.loads(line)
            ids, _ = train_sft.encode(tok, r["messages"], tools)
            yield r, len(ids)

    for src, dst, every in (("train.jsonl", "train.jsonl", 1),
                            ("eval.jsonl", "validation.jsonl", a.eval_every)):
        kept = [r for r, n in rows(src) if n <= a.max_len][::every]
        with open(out / dst, "w", encoding="utf-8") as fh:
            for r in kept:
                fh.write(json.dumps({"messages": r["messages"], "tools": tools},
                                    ensure_ascii=False) + "\n")
        print(f"{dst}: {len(kept)} rows")

    if not a.check:
        return
    sys.path.insert(0, a.studio)
    from common import conversation
    fmt = {"mode": "auto", "use_model_template": True, "chat_template": tok.chat_template,
           "specials": {k: v for k, v in (("bos_token", tok.bos_token),
                                          ("eos_token", tok.eos_token),
                                          ("pad_token", tok.pad_token),
                                          ("unk_token", tok.unk_token)) if v}}
    from common.formatting import detect_format
    first = json.loads(open(out / "train.jsonl", encoding="utf-8").readline())
    detected = detect_format(list(first.keys()), [first])
    detected.pop("confidence", None)
    fmt = {**detected, **{k: v for k, v in fmt.items() if k != "mode"}}
    print("format:", {k: v for k, v in fmt.items() if k not in ("chat_template", "specials")})
    same_ids = same_labels = inexact = 0
    shown = 0
    for i, line in enumerate(open(out / "train.jsonl", encoding="utf-8")):
        if i >= a.check:
            break
        row = json.loads(line)
        ids0, lab0 = train_sft.encode(tok, row["messages"], tools)
        ids1, lab1, exact = studio_encode(tok, row, fmt, conversation, a.max_len)
        inexact += not exact
        same_ids += ids0 == ids1
        same_labels += lab0 == lab1
        if (ids0 != ids1 or lab0 != lab1) and shown < 2:
            shown += 1
            t0 = tok.decode([t for t in lab0 if t != -100])
            t1 = tok.decode([t for t in lab1 if t != -100])
            print(f"--- row {i}: ids {'same' if ids0 == ids1 else 'differ'} "
                  f"({len(ids0)} vs {len(ids1)}), labels differ")
            if ids0 != ids1:
                k = next(j for j in range(min(len(ids0), len(ids1))) if ids0[j] != ids1[j])
                print("  first id difference at", k, repr(tok.decode(ids0[k - 20:k + 20])),
                      "||", repr(tok.decode(ids1[k - 20:k + 20])))
            print("  train_sft trains:", repr(t0[:400]))
            print("  ai-studio trains:", repr(t1[:400]))
    n = min(a.check, i + 1)
    print(f"checked {n}: same token ids {same_ids}, same trained tokens {same_labels}, "
          f"inexact spans {inexact}")


if __name__ == "__main__":
    main()
