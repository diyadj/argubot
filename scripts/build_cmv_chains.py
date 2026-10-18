"""Turn the CMV corpus into persona dialogues, split by thread (owner: Tricia).

    python scripts/build_cmv_chains.py

Input:  data/raw/convokit (from scripts/download_data.py --cmv)
Output: cfg["paths"]["cmv_chains"], one JSON dialogue per line, each carrying its split.
The real work is extract_persona_chains in src/argubot/data/cmv.py.
"""
import argparse
from pathlib import Path

from argubot.config import load_config
from argubot.data.cmv import extract_persona_chains
from argubot.io import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    from convokit import Corpus

    folder = Path(cfg["paths"]["raw_dir"]) / "convokit"
    found = sorted(folder.rglob("utterances.jsonl"))
    if not found:
        raise SystemExit(f"No ConvoKit corpus under {folder}. Run: python scripts/download_data.py --cmv")
    corpus = Corpus(filename=str(found[0].parent))

    rows = extract_persona_chains(corpus, cfg)
    n = write_jsonl(cfg["paths"]["cmv_chains"], rows)

    print(f"wrote {n} dialogues to {cfg['paths']['cmv_chains']}")
    for split in ("train", "val", "test"):
        part = [r for r in rows if r["split"] == split]
        threads = {r["dialogue_id"].rsplit("_", 1)[0] for r in part}
        bot = sum(t["speaker"] == "bot" for r in part for t in r["turns"])
        print(f"  {split:5s} {len(part):5d} dialogues from {len(threads):4d} threads, {bot} bot turns")


if __name__ == "__main__":
    main()
