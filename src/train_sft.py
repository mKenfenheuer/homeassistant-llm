"""LoRA SFT of Qwen2.5-3B-Instruct for Home Assistant tool calling, one model
per language (--lang, default de; paths default to data/<lang>/ and
output/<lang>/).

- Prompt/tool schema: custom_components/ha_llm_agent/llm_schema.py (shared
  with the integration, so training and runtime render identical prompts).
- Loss only on assistant tokens (answers and <tool_call> blocks).
- Tool-call arguments are passed as dicts: the Qwen2.5 template renders them
  with `tojson`, a JSON string would end up double-encoded.
- --init-adapter continues from the LoRA adapter of the last model instead
  of a fresh adapter (lower default lr 3e-5, 1 epoch; 1e-4 drove the eval loss of v18 up from the v17 level).
- The loss on every --eval-every-th sample of data/<lang>/eval.jsonl is
  measured every --eval-steps steps. An improvement overwrites the adapter in
  <out>/best (no other checkpoints are written); training stops when the
  loss has not improved by --min-delta (relative) for --patience
  evaluations, and the best adapter is the result.
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
from peft import (LoraConfig, PeftModel, get_peft_model, load_peft_weights,
                  set_peft_model_state_dict)
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainerCallback, TrainingArguments)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "custom_components" / "ha_llm_agent"))
import llm_schema  # noqa: E402

DEFAULTS = dict(
    model="Qwen/Qwen2.5-3B-Instruct",
    lang="de",
    data="data/{lang}/train.jsonl",
    eval_data="data/{lang}/eval.jsonl",
    out="output/{lang}/lora",
    merged="output/{lang}/merged",
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


class EvalLossStop(TrainerCallback):
    """Keep the best adapter and stop when the eval loss has not dropped by
    `min_delta` (relative to the best value so far) for `patience`
    evaluations in a row. Every new lowest loss overwrites <out>/best."""

    def __init__(self, patience: int, min_delta: float, best_dir: str):
        self.patience, self.min_delta, self.best_dir = patience, min_delta, best_dir
        self.best, self.bad, self.lowest, self.best_step = None, 0, None, None

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kw):
        loss = (metrics or {}).get("eval_loss")
        if loss is None:
            return
        if self.best is None or loss < self.best * (1 - self.min_delta):
            self.best, self.bad = loss, 0
        else:
            self.bad += 1
        saved = ""
        if model is not None and (self.lowest is None or loss < self.lowest):
            self.lowest, self.best_step = loss, state.global_step
            model.save_pretrained(self.best_dir)
            saved = ", saved as best"
        print(f"EVAL step {state.global_step}: eval_loss {loss:.5f}, best {self.best:.5f}, "
              f"no gain {self.bad}/{self.patience}{saved}", flush=True)
        if self.bad >= self.patience:
            print("EVAL: loss no longer drops, stopping", flush=True)
            control.should_training_stop = True


def main():
    ap = argparse.ArgumentParser()
    types = {"epochs": float, "batch": int, "accum": int, "max_len": int, "lr": float}
    for k in DEFAULTS:
        ap.add_argument("--" + k, default=None, type=types.get(k, str))
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--init-adapter", default=None,
                    help="LoRA adapter of the last model to continue from")
    ap.add_argument("--eval-steps", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=3,
                    help="use every n-th sample of the eval set (speed)")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--min-delta", type=float, default=0.02,
                    help="relative eval-loss gain that counts as progress")
    ap.add_argument("--resume", action="store_true",
                    help="continue from the newest checkpoint in --out")
    a = ap.parse_args()
    cfg = {k: getattr(a, k) if getattr(a, k) is not None else v
           for k, v in DEFAULTS.items()}
    if a.init_adapter:
        if a.lr is None:
            cfg["lr"] = 3e-5
        if a.epochs is None:
            cfg["epochs"] = 1
    for k in ("data", "eval_data", "out", "merged"):
        cfg[k] = cfg[k].format(lang=cfg["lang"])
    tools = llm_schema.build_tools(cfg["lang"])

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    def load(path):
        out, dropped = [], 0
        for l in open(path, encoding="utf-8"):
            ids, labels = encode(tok, json.loads(l)["messages"], tools)
            if len(ids) > cfg["max_len"]:
                dropped += 1
                continue
            out.append({"input_ids": ids, "labels": labels})
        return out, dropped

    enc, dropped = load(cfg["data"])
    eval_enc, _ = load(cfg["eval_data"])
    eval_enc = eval_enc[::a.eval_every]
    lens = sorted(len(e["input_ids"]) for e in enc)
    print(f"samples: {len(enc)} (dropped {dropped} > {cfg['max_len']} tokens), "
          f"tokens median {lens[len(lens)//2]}, p95 {lens[int(len(lens)*.95)]}, "
          f"max {lens[-1]}")
    # sanity: show what the loss sees for the first sample
    s = enc[0]
    print("TRAIN TARGET 0:", repr(tok.decode([t for t in s["labels"] if t != -100])))
    ds = Dataset.from_list(enc)
    eval_ds = Dataset.from_list(eval_enc)

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map={"": 0})
    if a.init_adapter:
        print("continuing from adapter", a.init_adapter)
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
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
        eval_strategy="steps",
        eval_steps=a.eval_steps,
        per_device_eval_batch_size=1,
        save_strategy="no",             # EvalLossStop keeps the best adapter
        dataloader_num_workers=2,
        report_to=[],
        seed=42,
    )
    print(f"steps: {total}, eval every {a.eval_steps} on {len(eval_ds)} samples, "
          f"lr {cfg['lr']}, epochs {cfg['epochs']}")
    trainer = Trainer(model=model, args=targs, train_dataset=ds, eval_dataset=eval_ds,
                      data_collator=collate_fn(pad_id),
                      callbacks=[stopper := EvalLossStop(a.patience, a.min_delta,
                                                         f"{cfg['out']}/best")])
    trainer.train(resume_from_checkpoint=True if a.resume else None)

    if stopper.best_step is not None:
        set_peft_model_state_dict(model, load_peft_weights(f"{cfg['out']}/best"))
        print(f"best checkpoint: step {stopper.best_step}, eval_loss {stopper.lowest:.5f}",
              flush=True)
    print("saving adapter ->", cfg["out"])
    model.save_pretrained(cfg["out"])
    merged = model.merge_and_unload()
    merged.save_pretrained(cfg["merged"], safe_serialization=True)
    tok.save_pretrained(cfg["merged"])
    print("merged ->", cfg["merged"])


if __name__ == "__main__":
    main()
