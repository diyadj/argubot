"""Print the config as a markdown table, ready to paste into the report's hyperparameter section.

    python scripts/print_hparams.py
"""
import argparse

from argubot.config import flatten, load_config

SKIP_PREFIXES = ("paths.",)  # file locations are not hyperparameters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    rows = [(k, v) for k, v in flatten(load_config(args.config)) if not k.startswith(SKIP_PREFIXES)]
    print("| Parameter | Value |")
    print("| --- | --- |")
    for key, value in rows:
        print(f"| {key} | {value} |")


if __name__ == "__main__":
    main()
