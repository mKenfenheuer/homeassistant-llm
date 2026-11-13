#!/usr/bin/env python3
"""Upload a language's AI Studio dataset (data/<lang>/studio, from
src/to_studio.py) and queue its fine-tune, continuing from an adapter run.

Usage: studio_lang_run.py LANG BASE_JOB [--lr 1e-4] [--project prj_...]
The API key (studio access) is read from ~/.config/ai-studio/key.
"""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("lang")
ap.add_argument("base_job")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--project", default="prj_0199ca9ac16a")
ap.add_argument("--studio", default="http://10.1.250.48")
ap.add_argument("--version", default="v18")
a = ap.parse_args()
key = (Path.home() / ".config/ai-studio/key").read_text().strip()
hdr = ["-H", "Authorization: Bearer " + key]
d = ROOT / "data" / a.lang / "studio"


def curl(*args):
    out = subprocess.run(["curl", "-sf", *hdr, *args], capture_output=True, text=True, check=True).stdout
    return json.loads(out)


name = "HA LLM %s %s" % (a.lang.upper(), a.version)
ds = curl("-X", "POST", "-F", "file=@%s;type=application/jsonl" % (d / "train.jsonl"),
          "%s/api/datasets/upload?name=%s&split=train&project=%s" % (a.studio, name.replace(" ", "%20"), a.project))
ds = curl("-X", "POST", "-F", "file=@%s;type=application/jsonl" % (d / "validation.jsonl"),
          "%s/api/datasets/upload?split=validation&into=%s" % (a.studio, ds["id"]))
print("dataset", ds["id"], ds["splits"])
run = {"kind": "finetune_llm", "name": "%s (from %s)" % (name, a.base_job), "project_id": a.project,
       "config": {"base_model_job": a.base_job, "studio_dataset": ds["id"],
                  "dataset_split": "train", "dataset_eval_split": "validation",
                  "format": {"mode": "auto", "use_model_template": True}, "train_on": "assistant",
                  "method": "lora", "quantization": "none", "dtype": "bfloat16", "max_seq_len": 4096,
                  "batch_size": 1, "grad_accum": 16, "epochs": 1, "learning_rate": a.lr,
                  "eval_every": 20, "early_stop": True, "early_stop_patience": 10,
                  "gradient_checkpointing": True, "optim_8bit": False, "merge_after": True,
                  "seed": 18}}
job = curl("-X", "POST", "-H", "content-type: application/json", "-d", json.dumps(run),
           "%s/api/jobs" % a.studio)
print("run", job["id"])
