"""Render training conversations as calibration text for llama-imatrix.

Usage: python src/make_calib.py data/train.jsonl model/calib.txt [N]

The importance matrix tells llama-quantize which weights matter for our
prompts (long device lists, tool calls), so Q4_K_M keeps them more exact.
Only training data is used, never the test set.
"""
from __future__ import annotations

import json
import random
import sys

from transformers import AutoTokenizer

sys.path.insert(0, "src")
from train_sft import llm_schema, to_template_msgs  # noqa: E402


def main():
    src, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]
    random.Random(0).shuffle(rows)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    with open(out, "w", encoding="utf-8") as f:
        for r in rows[:n]:
            f.write(tok.apply_chat_template(to_template_msgs(r["messages"]), tools=llm_schema.TOOLS,
                                            tokenize=False, add_generation_prompt=False))
    print(f"{n} conversations -> {out}")


if __name__ == "__main__":
    main()
