"""Queue a calibrated GGUF export of a language's fine-tune in AI Studio.

The importance matrix comes from the language's own training data, the chat
template from templates/chat_template.<lang>.jinja (default system prompt).

    python scripts/studio_export.py es job_... ds_... [--version v18]
"""
import argparse
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("lang")
ap.add_argument("source_job")
ap.add_argument("dataset")
ap.add_argument("--version", default="v18")
ap.add_argument("--studio", default="http://10.1.250.48")
ap.add_argument("--project", default="prj_0199ca9ac16a")
a = ap.parse_args()
key = (Path.home() / ".config/ai-studio/key").read_text().strip()
L = a.lang.upper()
name = f"qwen2.5-3b-ha-{a.version}"
body = {"kind": "export_gguf", "project_id": a.project,
        "name": f"HA LLM {L} {a.version} → GGUF Q4_K_M (imatrix)",
        "config": {"source_job": a.source_job, "quantize": "Q4_K_M",
                   "imatrix": {"studio_dataset": a.dataset, "split": "train", "rows": 300,
                               "ctx": 4096, "seed": 0},
                   "chat_template": (ROOT / f"templates/chat_template.{a.lang}.jinja").read_text(),
                   "general_name": f"Qwen2.5 3B Home Assistant {L} {name}",
                   "name_hint": name, "what": "model"}}
req = urllib.request.Request(a.studio + "/api/jobs", data=json.dumps(body).encode(), method="POST",
                             headers={"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json"})
print("run", json.load(urllib.request.urlopen(req))["id"])
