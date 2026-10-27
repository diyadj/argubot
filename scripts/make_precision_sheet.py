"""How many of the strict metric's flags are real contradictions? A blinded sheet to find out.

    python scripts/make_precision_sheet.py            # after evaluate.py --human --strict
    python scripts/make_precision_sheet.py --score    # after the sheet is filled in

Takes up to --n flagged pairs per system and from the real CMV replies (one pair per flagged turn,
so a long debate cannot dominate), shuffles them and writes precision_sheet.csv with no system
names, plus precision_key_PRIVATE.csv. The labeller marks real_contradiction 1 or 0 per pair.
It must be someone who is not a human rater (Tricia, Cassey or Andrea), since the pairs quote
bot replies. --score prints precision per system and the per-turn rate multiplied by it.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

from argubot.config import ensure_dir, load_config
from argubot.io import read_jsonl

SOURCES = ["A", "B", "C1", "C2", "D", "human"]


def make(cfg, n: int) -> None:
    results = Path(cfg["paths"]["results_dir"])
    rng = random.Random(cfg["seed"])
    rows = []
    for source in SOURCES:
        by_turn = {}
        for f in read_jsonl(str(results / f"flagged_strict_{source}.jsonl")):
            by_turn.setdefault((f["dialogue_id"], f["turn"]), []).append(f)
        turns = sorted(by_turn)
        for key in rng.sample(turns, min(n, len(turns))):
            rows.append((source, rng.choice(by_turn[key])))
    rng.shuffle(rows)

    out = ensure_dir(cfg["paths"]["rating_dir"])
    with open(out / "precision_key_PRIVATE.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "system", "dialogue_id", "turn", "score"])
        for i, (source, p) in enumerate(rows):
            w.writerow([f"pair{i:03d}", source, p["dialogue_id"], p["turn"], f"{p['score']:.3f}"])
    with open(out / "precision_sheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "earlier_claim", "new_sentence", "real_contradiction", "notes"])
        for i, (_, p) in enumerate(rows):
            w.writerow([f"pair{i:03d}", p["earlier_claim"], p["new_sentence"], "", ""])
    print(f"wrote {len(rows)} pairs to {out / 'precision_sheet.csv'} (key: precision_key_PRIVATE.csv)")


def score(cfg) -> None:
    out = Path(cfg["paths"]["rating_dir"])
    with open(out / "precision_key_PRIVATE.csv", newline="", encoding="utf-8") as f:
        key = {r["pair_id"]: r["system"] for r in csv.DictReader(f)}
    with open(out / "precision_sheet.csv", newline="", encoding="utf-8") as f:
        labels = {r["pair_id"]: r["real_contradiction"].strip() for r in csv.DictReader(f)}
    bad = [i for i, v in labels.items() if v not in ("0", "1")]
    if bad:
        sys.exit(f"{len(bad)} pairs are not marked 0 or 1, e.g. {bad[:5]}")
    metrics = json.loads((Path(cfg["paths"]["results_dir"]) / "metrics_strict.json").read_text())

    print(f"{'Source':<8}{'Labelled':>9}{'Real':>6}{'Precision':>10}{'Per turn':>10}{'x precision':>12}")
    summary = {}
    for source in SOURCES:
        marks = [int(labels[i]) for i, s in key.items() if s == source]
        if not marks:
            continue
        precision = sum(marks) / len(marks)
        rate = metrics[source]["contradiction_rate"]
        summary[source] = {"labelled": len(marks), "real": sum(marks), "precision": precision, "corrected_rate": rate * precision}
        print(f"{source:<8}{len(marks):>9}{sum(marks):>6}{precision:>10.2f}{rate:>10.3f}{rate * precision:>12.3f}")
    path = ensure_dir(cfg["paths"]["results_dir"]) / "precision_strict.json"
    path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved to {path}. 'x precision' is approximate: it assumes a flagged turn's other pairs are as real as the one labelled.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--n", type=int, default=30, help="pairs per system")
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    score(cfg) if args.score else make(cfg, args.n)


if __name__ == "__main__":
    main()
