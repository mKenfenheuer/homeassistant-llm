"""Upload the GGUF model and the dataset to Hugging Face.

Requires `hf auth login` (write token) on this machine.
Usage: python src/upload_hf.py MODEL_REPO DATASET_REPO [--private]
  e.g. python src/upload_hf.py maxk/qwen2.5-3b-homeassistant-de-GGUF \
                               maxk/homeassistant-tool-calling-de
"""

import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
GGUF = ROOT / "model" / "qwen2.5-3b-ha-Q4_K_M.gguf"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    model_repo, dataset_repo = sys.argv[1], sys.argv[2]
    private = "--private" in sys.argv
    api = HfApi()
    print("logged in as", api.whoami()["name"])
    digest = sha256(GGUF)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # ---- model repo
        card = (ROOT / "hf/model/README.md").read_text()
        card = card.replace("__DATASET_REPO__", dataset_repo).replace("__SHA256__", digest)
        (tmp / "model").mkdir()
        (tmp / "model/README.md").write_text(card)
        shutil.copy(ROOT / "hf/model/NOTICE", tmp / "model/NOTICE")
        lic = hf_hub_download("Qwen/Qwen2.5-3B-Instruct", "LICENSE")
        shutil.copy(lic, tmp / "model/LICENSE")
        (tmp / "model" / GGUF.name).symlink_to(GGUF)
        api.create_repo(model_repo, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(repo_id=model_repo, repo_type="model", folder_path=tmp / "model",
                          commit_message="Upload qwen2.5-3b-ha Q4_K_M")
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
