"""Paired differences between systems, and between the two models, from the human ratings.

    python scripts/rating_differences.py

Reads the filled sheets in data/human_ratings/<run>/ (rater_1.csv, rater_2.csv, key_PRIVATE.csv).
Each debate's score for a system is the mean of the two raters, and every interval is a paired 95%
bootstrap over the 20 debates, the same method as the contradiction tables in docs/RESULTS.md.
"""
import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple

from argubot.eval.human_eval import CRITERIA

SYSTEMS = ["A", "B", "C1", "C2", "D"]
STEPS = [("B", "A"), ("C1", "B"), ("C2", "C1"), ("D", "C2"), ("D", "B")]


def _rows(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def debate_scores(run_dir: Path) -> Dict[Tuple[str, str, str], float]:
    """(system, dialogue_id, criterion) -> mean score over the raters."""
    key = {r["item_id"]: r for r in _rows(run_dir / "key_PRIVATE.csv")}
    sums: Dict[Tuple[str, str, str], List[int]] = {}
    for sheet in sorted(run_dir.glob("rater_*.csv")):
        for row in _rows(sheet):
            k = key[row["item_id"]]
            for crit in CRITERIA:
                sums.setdefault((k["system"], k["dialogue_id"], crit), []).append(int(row[crit].strip()))
    return {k: sum(v) / len(v) for k, v in sums.items()}


def paired_ci(diffs: List[float], n_boot: int = 10000, seed: int = 0) -> Tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    boots = sorted(sum(rng.choice(diffs) for _ in range(n)) / n for _ in range(n_boot))
    return sum(diffs) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def fmt(m: float, lo: float, hi: float) -> str:
    star = " *" if lo > 0 or hi < 0 else ""
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/human_ratings")
    ap.add_argument("--runs", nargs=2, default=["0.8b", "4b"])
    args = ap.parse_args()

    scores = {run: debate_scores(Path(args.dir) / run) for run in args.runs}
    dids = sorted({k[1] for k in scores[args.runs[0]]})

    for run in args.runs:
        s = scores[run]
        print(f"\n{run}: paired differences between systems (* = interval excludes zero)")
        print(f"{'Step':<10}" + "".join(f"{c:>30}" for c in CRITERIA))
        for a, b in STEPS:
            cells = [fmt(*paired_ci([s[(a, d, c)] - s[(b, d, c)] for d in dids])) for c in CRITERIA]
            print(f"{a + ' - ' + b:<10}" + "".join(f"{x:>30}" for x in cells))

    lo_run, hi_run = args.runs
    print(f"\n{hi_run} minus {lo_run}, averaged over the five systems (different rater pairs)")
    for c in CRITERIA:
        diffs = [
            sum(scores[hi_run][(sy, d, c)] - scores[lo_run][(sy, d, c)] for sy in SYSTEMS) / len(SYSTEMS)
            for d in dids
        ]
        print(f"{c:<26}{fmt(*paired_ci(diffs)):>30}")


if __name__ == "__main__":
    main()
