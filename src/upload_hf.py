"""Upload a model and/or the dataset of one language to Hugging Face.

All languages share one model repository and one dataset repository:
  model:   models.json (language -> version, file, sha256, schema; read by the
           add-on), <lang>/<model>.gguf, <lang>/chat_template.jinja, README.md,
           LICENSE, NOTICE. latest.json and the v15.1 GGUF at the top level
           stay for add-ons up to 0.11 (German, schema 5).
  dataset: train.jsonl and eval.jsonl with a "locale" column for all
           languages, the generator (gen_dataset.py, gen/<lang>/) and
           llm_schema.py
Uploading one language replaces only its own files and rows; --all (or
--langs de,en,...) uploads the dataset of several languages at once, as
written by src/gen_all.py.

Requires HF_TOKEN in the environment (or `hf auth login`) with write access.
Usage: python src/upload_hf.py [--lang de] [--gguf FILE] [--dataset-only]
                               [--model-repo ID] [--dataset-repo ID] [--private]
       python src/upload_hf.py --all | --langs de,en,...   (dataset only)
"""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPO = "mkenfenheuer/qwen2.5-3b-homeassistant-GGUF"
DATASET_REPO = "mkenfenheuer/homeassistant-tool-calling"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def opt(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        val = argv[i + 1]
        del argv[i:i + 2]
        return val
    return default


def upload_model(api, tmp, gguf, lang, model_repo, dataset_repo, private):
    digest = sha256(gguf)
    hf = ROOT / "hf/model"
    # models.json for the add-on's automatic update: hf/model/<lang>/latest.json
    # holds version, schema and notes; file, hash and size come from the GGUF.
    # The entries of the other languages are kept.
    manifest = json.loads((hf / lang / "latest.json").read_text())
    try:   # the entry of the released model before this one
        manifest_prev = json.loads(Path(hf_hub_download(model_repo, "models.json")).read_text()
                                   )["languages"].get(lang, {})
    except (EntryNotFoundError, RepositoryNotFoundError, KeyError, ValueError):
        manifest_prev = {}
    if manifest_prev.get("file") == f"{lang}/{gguf.name}":
        manifest_prev = {}
    manifest.update(file=f"{lang}/{gguf.name}", sha256=digest, size=gguf.stat().st_size)
    index = {"languages": {}}
    try:
        index = json.loads(Path(hf_hub_download(model_repo, "models.json")).read_text())
    except (EntryNotFoundError, RepositoryNotFoundError):
        pass
    index.setdefault("languages", {})[lang] = manifest
    index["languages"] = dict(sorted(index["languages"].items()))
    out = tmp / "model"
    (out / lang).mkdir(parents=True)
    card = (hf / "README.md").read_text()
    (out / "README.md").write_text(card.replace("__DATASET_REPO__", dataset_repo)
                                   .replace("__MODEL_REPO__", model_repo))
    (out / "models.json").write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n")
    shutil.copy(hf / "NOTICE", out / "NOTICE")
    shutil.copy(ROOT / f"templates/chat_template.{lang}.jinja", out / lang / "chat_template.jinja")
    shutil.copy(hf_hub_download("Qwen/Qwen2.5-3B-Instruct", "LICENSE"), out / "LICENSE")
    (out / lang / gguf.name).symlink_to(gguf)
    api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)
    # the current GGUF of this language and the previous one stay (the
    # previous add-on version still downloads it as its default model);
    # older ones are deleted in the same commit, other languages untouched
    keep = {f"{lang}/{gguf.name}"}
    if prev := manifest_prev.get("file"):
        keep.add(prev)
    old = [f for f in api.list_repo_files(model_repo)
           if f.startswith(f"{lang}/") and f.endswith(".gguf") and f not in keep]
    api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=out,
                      commit_message=f"Modell {lang} {manifest.get('version', gguf.stem)}",
                      delete_patterns=old or None)
    print(f"model:     https://huggingface.co/{model_repo}")
    print(f"model_url: https://huggingface.co/{model_repo}/resolve/main/{lang}/{gguf.name}")
    print(f"sha256:    {digest}")


def merged_rows(api, repo, fname, langs):
    """Rows of the other languages from the hub plus the local rows of `langs`."""
    rows = []
    try:
        path = hf_hub_download(repo, fname, repo_type="dataset")
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    except (EntryNotFoundError, RepositoryNotFoundError):
        pass
    # rows uploaded before the locale column were German
    rows = [r for r in rows if r.get("locale", "de") not in langs]
    for lang in langs:
        for line in open(ROOT / "data" / lang / fname, encoding="utf-8"):
            r = json.loads(line)
            rows.append({"locale": lang, **{k: v for k, v in r.items() if k != "locale"}})
    rows.sort(key=lambda r: r["locale"])
    return rows


def split_table(counts) -> str:
    locales = sorted({l for c in counts.values() for l in c})
    lines = ["| Locale | train | test |", "|--------|-------|------|"]
    for l in locales:
        lines.append(f"| {l} | {counts['train.jsonl'].get(l, 0):,} | "
                     f"{counts['eval.jsonl'].get(l, 0):,} |")
    return "\n".join(lines)


def upload_dataset(api, tmp, langs, dataset_repo, private):
    out = tmp / "dataset"
    (out / "gen").mkdir(parents=True)
    api.create_repo(dataset_repo, repo_type="dataset", private=private, exist_ok=True)
    counts = {}
    for f in ("train.jsonl", "eval.jsonl"):
        rows = merged_rows(api, dataset_repo, f, langs)
        with open(out / f, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[f] = {}
        for r in rows:
            counts[f][r["locale"]] = counts[f].get(r["locale"], 0) + 1
    card = (ROOT / "hf/dataset/README.md").read_text()
    (out / "README.md").write_text(card.replace("__SPLITS__", split_table(counts)))
    # generator: entry point, shared package file and the packages of the languages
    shutil.copy(ROOT / "src/gen_dataset.py", out / "gen_dataset.py")
    shutil.copy(ROOT / "src/gen_all.py", out / "gen_all.py")
    shutil.copy(ROOT / "src/gen/__init__.py", out / "gen/__init__.py")
    for lang in langs:
        (out / "gen" / lang).mkdir()
        for py in (ROOT / "src/gen" / lang).glob("*.py"):
            shutil.copy(py, out / "gen" / lang / py.name)
    shutil.copy(ROOT / "custom_components/ha_llm_agent/llm_schema.py", out / "llm_schema.py")
    api.upload_folder(repo_id=dataset_repo, repo_type="dataset", folder_path=out,
                      commit_message=f"Datensatz {', '.join(langs)}")
    print(f"dataset:   https://huggingface.co/datasets/{dataset_repo}  rows per locale: {counts}")


def main():
    argv = sys.argv[1:]
    lang = opt(argv, "--lang", "de")
    gguf = opt(argv, "--gguf")
    model_repo = opt(argv, "--model-repo", MODEL_REPO)
    dataset_repo = opt(argv, "--dataset-repo", DATASET_REPO)
    private = "--private" in argv
    dataset_only = "--dataset-only" in argv
    langs = [lang]
    if "--all" in argv:
        langs = sorted(p.name for p in (ROOT / "data").iterdir()
                       if p.name != "all" and (p / "train.jsonl").exists()
                       and (ROOT / "src/gen" / p.name).is_dir())
        dataset_only = True
    elif "--langs" in argv:
        langs = opt(argv, "--langs").split(",")
        dataset_only = True
    if not dataset_only and not gguf:
        sys.exit("--gguf FILE is required unless --dataset-only, --all or --langs is given")
    api = HfApi()
    print("logged in as", api.whoami()["name"], "- languages:", ", ".join(langs))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        if not dataset_only:
            upload_model(api, tmp, Path(gguf).resolve(), lang, model_repo, dataset_repo, private)
        upload_dataset(api, tmp, langs, dataset_repo, private)


if __name__ == "__main__":
    main()
