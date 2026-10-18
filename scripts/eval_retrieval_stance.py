"""Retrieval-stance test on the 71 IBM-ArgQ motions (owner: Diya).

    python scripts/eval_retrieval_stance.py                      # BM25, every argument as a query, about 2 minutes on a CPU
    python scripts/eval_retrieval_stance.py --method dense       # MiniLM built in memory over the 30k short arguments
    python scripts/eval_retrieval_stance.py --max-queries 2000   # a seeded sample, faster

What the numbers mean is in src/argubot/eval/retrieval_stance.py. Writes
cfg["paths"]["results_dir"]/retrieval_stance_<method>.json, including one row per motion.
"""
import argparse
import json
import time
from pathlib import Path

from argubot.config import load_config
from argubot.eval.retrieval_stance import argument_query_test, motion_query_test
from argubot.io import read_jsonl
from argubot.retrieval.bm25 import BM25Retriever
from argubot.schemas import Passage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--method", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-queries", type=int, default=None, help="seeded sample of argument queries (default: all)")
    args = ap.parse_args()
    cfg = load_config(args.config)

    passages = [Passage.from_dict(d) for d in read_jsonl(cfg["paths"]["passages_backup"])]
    start = time.time()
    if args.method == "bm25":
        retriever = BM25Retriever(passages)
    else:
        from argubot.retrieval.dense import DenseRetriever

        retriever = DenseRetriever(cfg["retrieval"]["dense_model"])
        retriever.build(passages, on_progress=lambda d, t: print(f"  encoded {d}/{t}", flush=True))
    print(f"{args.method} index over {len(passages)} IBM-ArgQ passages ready in {time.time() - start:.0f}s")

    motions = motion_query_test(retriever, passages, args.k)
    hit = sum(m["hit_at_k"] for m in motions) / len(motions)
    pro = [m["pro_share"] for m in motions if m["pro_share"] is not None]
    print(f"\nTest 1, motion as query ({len(motions)} motions, top {args.k}):")
    print(f"  motion hit@{args.k}: {hit:.2f}   PRO share of hits: {sum(pro) / len(pro):.2f} (0.50 is balanced)")
    worst = sorted(motions, key=lambda m: m["hit_at_k"])[:3]
    print("  weakest motions:", "; ".join(f"{m['motion']} ({m['hit_at_k']:.1f})" for m in worst))

    start = time.time()
    r = argument_query_test(retriever, passages, args.k, args.max_queries, cfg["seed"])
    print(f"\nTest 2, argument as query ({r['n_queries']} queries, {time.time() - start:.0f}s):")
    print(f"  topical hit@{args.k}:          {r['topical_hit_at_k']:.3f}")
    print(f"  same-stance rate:        {r['same_stance_rate']:.3f}")
    print(f"  chance:                  {r['chance_same_stance_rate']:.3f}")
    print(f"  gap over chance:         {r['gap_over_chance']:+.3f}  (95% CI {r['gap_ci_low']:+.3f} to {r['gap_ci_high']:+.3f})")

    out = Path(cfg["paths"]["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"retrieval_stance_{args.method}.json"
    path.write_text(json.dumps({"method": args.method, "per_motion": motions, "argument_query": r}, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
