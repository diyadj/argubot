"""Turn the CMV corpus into persona dialogues, split by thread (owner: Tricia).

    python scripts/build_cmv_chains.py

Input:  data/raw/convokit (from scripts/download_data.py --cmv)
Output: cfg["paths"]["cmv_chains"], one JSON dialogue per line, each carrying its split.
The real work is extract_persona_chains in src/argubot/data/cmv.py.
"""
import argparse

from argubot.config import load_config
from argubot.data.cmv import extract_persona_chains


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # TODO (Tricia): load the ConvoKit corpus from cfg["paths"]["raw_dir"], call
    # extract_persona_chains(corpus, cfg), and write the result with argubot.io.write_jsonl.
    extract_persona_chains(None, cfg)


if __name__ == "__main__":
    main()
