"""Train the LoRA adapter (owner: Andrea).

    python scripts/train_lora.py

The pieces are in src/argubot/generate/training.py. The step-by-step TODO is in train().
In Colab, the same call goes in the training notebook instead.
"""
import argparse

from argubot.config import load_config
from argubot.generate.training import train


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
