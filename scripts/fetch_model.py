"""Download the base model named in a config into the Hugging Face cache, once, before any GPU job.

    python scripts/fetch_model.py --config configs/config_4b.yaml

Run it on the cluster's login node with the group venv active (docs/UPGRADE_TO_4B.md). The cache
lives in the group account's home (~/.cache/huggingface), so every job and every teammate on the
CS425G3 login reuses this one copy. Running it again only checks that everything is there.
"""
import argparse
import os
from pathlib import Path

from argubot.config import load_config


def folder_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.realpath(os.path.join(root, name)))
    return total / 1024**3


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config_4b.yaml")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    name = load_config(args.config)["generator"]["base_model"]
    print(f"Fetching {name} (the 4B model is about 9 GB, this can take a while on the cluster)...", flush=True)
    path = snapshot_download(name)
    print(f"Downloaded to {path} ({folder_size_gb(path):.1f} GB)")

    config = AutoConfig.from_pretrained(name)
    text = getattr(config, "text_config", config)
    tok = AutoTokenizer.from_pretrained(name)
    print(f"Loads fine: {getattr(text, 'num_hidden_layers', '?')} layers, hidden size "
          f"{getattr(text, 'hidden_size', '?')}, tokenizer with {len(tok):,} tokens")
    print(f"HF cache: {os.environ.get('HF_HOME') or Path.home() / '.cache' / 'huggingface'}")


if __name__ == "__main__":
    main()
