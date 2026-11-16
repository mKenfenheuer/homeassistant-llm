#!/usr/bin/env python3
"""Upload a language's AI Studio dataset (data/<lang>/studio, from
src/to_studio.py) and queue its fine-tune, continuing from an adapter run.

Usage: studio_lang_run.py LANG BASE_JOB [--lr 1e-4] [--project prj_...]
The API key (studio access) is read from ~/.config/ai-studio/key.
"""
import argparse
import atexit
import json
import os
import subprocess
import sys
import tempfile
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
d = ROOT / "data" / a.lang / "studio"
# The key goes to curl as a header file, never on the command line: a failed
# call prints its arguments, and the key must not end up in a log.
_hfile = tempfile.NamedTemporaryFile("w", delete=False, prefix="studio-hdr-")
os.chmod(_hfile.name, 0o600)
_hfile.write("Authorization: Bearer %s\n"
             % (Path.home() / ".config/ai-studio/key").read_text().strip())
_hfile.close()
atexit.register(os.unlink, _hfile.name)


def curl(*args):
    res = subprocess.run(["curl", "-s", "-w", "\n%{http_code}", "-H", "@" + _hfile.name, *args],
                         capture_output=True, text=True)
    body, _, code = res.stdout.rpartition("\n")
    if res.returncode or not code.startswith("2"):
        sys.exit("studio call failed (HTTP %s): %s" % (code, body[:300]))
    return json.loads(body)


name = "HA LLM %s %s" % (a.lang.upper(), a.version)


def parts(path, limit=200 * 1024 * 1024):
    """The studio takes at most 256 MB per upload: cut the file at row ends."""
    out, buf, size = [], [], 0
    for line in open(path, "rb"):
        if size + len(line) > limit and buf:
            out.append(buf)
            buf, size = [], 0
        buf.append(line)
        size += len(line)
    out.append(buf)
    files = []
    for i, rows in enumerate(out):
        f = tempfile.NamedTemporaryFile("wb", delete=False, suffix=".jsonl", prefix="part%d-" % i)
        f.writelines(rows)
        f.close()
        atexit.register(os.unlink, f.name)
        files.append(f.name)
    return files


train_parts = parts(d / "train.jsonl")
ds = curl("-X", "POST", "-F", "file=@%s;type=application/jsonl" % train_parts[0],
          "%s/api/datasets/upload?name=%s&split=train&project=%s" % (a.studio, name.replace(" ", "%20"), a.project))
for part in train_parts[1:]:
    ds = curl("-X", "POST", "-F", "file=@%s;type=application/jsonl" % part,
              "%s/api/datasets/upload?split=train&into=%s" % (a.studio, ds["id"]))
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
