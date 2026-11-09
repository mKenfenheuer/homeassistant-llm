"""Upload the GGUF model and the dataset to Hugging Face.

Requires `hf auth login` (write token) on this machine.
Usage: python src/upload_hf.py MODEL_REPO DATASET_REPO [--gguf FILE] [--private]
  e.g. python src/upload_hf.py maxk/qwen2.5-3b-homeassistant-de-GGUF \
                               maxk/homeassistant-tool-calling-de
"""

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "model" / "qwen2.5-3b-ha-Q4_K_M.gguf"   # default, see --gguf


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    global GGUF
    model_repo, dataset_repo = sys.argv[1], sys.argv[2]
    private = "--private" in sys.argv
    if "--gguf" in sys.argv:
        GGUF = Path(sys.argv[sys.argv.index("--gguf") + 1]).resolve()
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    digest = sha256(GGUF)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- model repo
        # manifest for the add-on's automatic update (hf/model/latest.json
        # holds version, schema and notes; file, hash and size come from GGUF)
        manifest = json.loads((ROOT / "hf/model/latest.json").read_text())
        manifest.update(file=GGUF.name, sha256=digest, size=GGUF.stat().st_size)
        manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
        card = (ROOT / "hf/model/README.md").read_text()
        card = (card.replace("__DATASET_REPO__", dataset_repo)
                .replace("__SHA256__", digest).replace("__GGUF__", GGUF.name)
                .replace("__MANIFEST__", manifest_text.strip()))
        (tmp / "model").mkdir()
        (tmp / "model/README.md").write_text(card)
        (tmp / "model/latest.json").write_text(manifest_text)
        shutil.copy(ROOT / "hf/model/NOTICE", tmp / "model/NOTICE")
        shutil.copy(ROOT / "templates/chat_template.jinja", tmp / "model/chat_template.jinja")
        lic = hf_hub_download("Qwen/Qwen2.5-3B-Instruct", "LICENSE")
        shutil.copy(lic, tmp / "model/LICENSE")
        (tmp / "model" / GGUF.name).symlink_to(GGUF)
        api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)
        # only the current GGUF stays in the repository; older files are
        # deleted in the same commit (uploaded files are never deleted)
        api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=tmp / "model",
                          commit_message=f"Modell {manifest.get('version', GGUF.stem)}",
                          delete_patterns=["*.gguf"])
        # ---- dataset repo
        (tmp / "dataset").mkdir()
        shutil.copy(ROOT / "hf/dataset/README.md", tmp / "dataset/README.md")
        for f in ("train.jsonl", "eval.jsonl"):
            shutil.copy(ROOT / "data" / f, tmp / "dataset" / f)
        shutil.copy(ROOT / "src/gen_dataset.py", tmp / "dataset/gen_dataset.py")
        shutil.copy(ROOT / "custom_components/ha_llm_agent/llm_schema.py",
                    tmp / "dataset/llm_schema.py")
        api.create_repo(dataset_repo, repo_type="dataset", private=private, exist_ok=True)
        api.upload_folder(repo_id=dataset_repo, repo_type="dataset",
                          folder_path=tmp / "dataset", commit_message="Upload dataset")

    print(f"model:   https://huggingface.co/{model_repo}")
    print(f"dataset: https://huggingface.co/datasets/{dataset_repo}")
    print(f"model_url: https://huggingface.co/{model_repo}/resolve/main/{GGUF.name}")
    print(f"sha256:  {digest}")


if __name__ == "__main__":
    main()
