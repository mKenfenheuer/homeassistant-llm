"""Merge + dedupe the template and teacher JSONL into one training file."""
import json
import random
import sys


def norm(s):
    return "".join(s.lower().split())


def main():
    seen = set()
    rows = []
    for path in INPUTS:
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    key = norm(json.dumps(
                        [m.get("content") or "" for m in r["messages"]],
                        ensure_ascii=False))
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(r)
        except FileNotFoundError:
            print("skip missing", path)
    random.Random(3).shuffle(rows)
    out = sys.argv[1] if len(sys.argv) > 1 else "data/train.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("merged", len(rows), "rows ->", out)


INPUTS = ["data/template.jsonl", "data/teacher.jsonl"]

if __name__ == "__main__":
    main()
