"""Pick retrieval.backup_min_score from real queries (owner: Diya).

    python scripts/calibrate_backup.py --percentile 10

Takes a seeded sample of the training queries (latest user turn plus claim log, exactly what the
pipeline sends), looks at the best args.me score for each, and prints the score below which the
chosen share of queries would fall back to IBM-ArgQ. Copy that number into
retrieval.backup_min_score. BM25 scores depend on query length, so recalibrate if the query
format changes, and again for dense retrieval, whose cosine scores have a different scale.
"""
import argparse
import random

import numpy as np

from argubot.config import load_config
from argubot.io import read_jsonl
from argubot.retrieval.base import build_query
from argubot.schemas import Passage
from argubot.systems.components import build_retriever


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--percentile", type=float, default=10.0, help="share of queries that should fall back")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["retrieval"]["use_backup"] = False  # measure the primary on its own

    examples = list(read_jsonl(cfg["paths"]["train_examples"]))
    sample = random.Random(cfg["seed"]).sample(examples, min(args.n, len(examples)))
    retriever = build_retriever(cfg, [Passage.from_dict(d) for d in read_jsonl(cfg["paths"]["passages"])])

    scores = np.array(
        [retriever.retrieve(build_query(e["history"][-1]["text"], e["claims"]), k=1)[0].score for e in sample]
    )
    print(f"best args.me score over {len(scores)} real queries:")
    for p in (1, 5, 10, 25, 50, 75, 90):
        print(f"  {p:>2}th percentile: {np.percentile(scores, p):6.1f}")
    print(f"\nbackup_min_score for {args.percentile:g}% fallback: {np.percentile(scores, args.percentile):.1f}")


if __name__ == "__main__":
    main()
