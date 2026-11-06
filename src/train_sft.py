"""LoRA SFT of Qwen2.5-3B-Instruct for German Home Assistant tool calling.

- Prompt/tool schema: custom_components/ha_llm_agent/llm_schema.py (shared
  with the integration, so training and runtime render identical prompts).
- Loss only on assistant tokens (answers and <tool_call> blocks).
- Tool-call arguments are passed as dicts: the Qwen2.5 template renders them
  with `tojson`, a JSON string would end up double-encoded.
- After training, the adapter is merged and saved as bf16 to --merged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "custom_components" / "ha_llm_agent"))
import llm_schema  # noqa: E402

DEFAULTS = dict(
    model="Qwen/Qwen2.5-3B-Instruct",
    data="data/train.jsonl",
    out="output/lora",
    merged="output/merged",
    epochs=2, batch=1, accum=16, lr=2e-4, max_len=4096,
)


def to_template_msgs(msgs):
    out = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {**tc, "function": {**tc["function"],
                                    "arguments": json.loads(tc["function"]["arguments"])}}
                for tc in m["tool_calls"]]
        out.append(m)
    return out


def encode(tok, msgs, tools):
    msgs = to_template_msgs(msgs)

    def ids_of(k):
        text = tok.apply_chat_template(msgs[:k], tools=tools, tokenize=False,
                                       add_generation_prompt=False)
        return tok(text, add_special_tokens=False)["input_ids"]

    ids = ids_of(len(msgs))
    labels = [-100] * len(ids)
    prev = len(ids_of(1))
    for i in range(1, len(msgs)):
        end = len(ids_of(i + 1))
        if msgs[i]["role"] == "assistant":
            # skip the "<|im_start|>assistant\n" header (3 tokens)
            for j in range(prev + 3, end):
                labels[j] = ids[j]
        prev = end
    return ids, labels


def collate_fn(pad_id):
    def collate(features):
        n = max(len(f["input_ids"]) for f in features)
        ids, labs, att = [], [], []
        for f in features:
            k = n - len(f["input_ids"])
            ids.append(f["input_ids"] + [pad_id] * k)
            labs.append(f["labels"] + [-100] * k)
            att.append([1] * len(f["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
                "attention_mask": torch.tensor(att)}
    return collate


def main():
    ap = argparse.ArgumentParser()
    types = {"epochs": float, "batch": int, "accum": int, "max_len": int, "lr": float}
    for k in DEFAULTS:
        ap.add_argument("--" + k, default=None, type=types.get(k, str))
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in --out")
    a = ap.parse_args()
    cfg = {k: getattr(a, k) if getattr(a, k) is not None else v
           for k, v in DEFAULTS.items()}

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = [json.loads(l) for l in open(cfg["data"], encoding="utf-8")]
    enc, dropped = [], 0
    for r in rows:
        ids, labels = encode(tok, r["messages"], llm_schema.TOOLS)
        if len(ids) > cfg["max_len"]:
            dropped += 1
            continue
        enc.append({"input_ids": ids, "labels": labels})
    lens = sorted(len(e["input_ids"]) for e in enc)
    print(f"samples: {len(enc)} (dropped {dropped} > {cfg['max_len']} tokens), "
          f"tokens median {lens[len(lens)//2]}, p95 {lens[int(len(lens)*.95)]}, "
          f"max {lens[-1]}")
    # sanity: show what the loss sees for the first sample
    s = enc[0]
    print("TRAIN TARGET 0:", repr(tok.decode([t for t in s["labels"] if t != -100])))
    ds = Dataset.from_list(enc)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0})
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    steps_per_epoch = math.ceil(len(ds) / (cfg["batch"] * cfg["accum"]))
    total = a.max_steps if a.max_steps > 0 else int(steps_per_epoch * cfg["epochs"])
    targs = TrainingArguments(
        output_dir=cfg["out"],
        remove_unused_columns=False,
        max_steps=a.max_steps,
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch"],
        gradient_accumulation_steps=cfg["accum"],
        learning_rate=cfg["lr"],
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.03 * total)),
        weight_decay=0.0,
        bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(50, total // 4),
        save_total_limit=2,
        dataloader_num_workers=2,
        report_to=[],
        seed=42,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collate_fn(pad_id))
    trainer.train(resume_from_checkpoint=True if a.resume else None)

    print("saving adapter ->", cfg["out"])
    model.save_pretrained(cfg["out"])
    merged = model.merge_and_unload()
    merged.save_pretrained(cfg["merged"], safe_serialization=True)
    tok.save_pretrained(cfg["merged"])
    print("merged ->", cfg["merged"])


if __name__ == "__main__":
    main()
