"""Turn the two filled-in rating sheets into kappa and per-system scores (owner: Cassey).

    python scripts/score_ratings.py

Reads rater_1.csv, rater_2.csv and key_PRIVATE.csv from paths.rating_dir (put the returned sheets
there under those names). Checks the sheets first: blanks and values other than 0 or 1 are listed,
and the script stops unless --allow-problems is given, since they would quietly skew the kappa.
Writes human_eval.json to paths.results_dir and prints two tables for the report.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

from argubot.config import ensure_dir, load_config
from argubot.eval.human_eval import CRITERIA, check_sheet, kappa_per_criterion, mean_scores_by_system


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--dir", default=None, help="folder with the sheets (default: paths.rating_dir)")
    ap.add_argument("--allow-problems", action="store_true", help="score anyway, skipping blank cells")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rdir = Path(args.dir or cfg["paths"]["rating_dir"])
    key = str(rdir / "key_PRIVATE.csv")
    sheets = [str(rdir / f"rater_{r + 1}.csv") for r in range(cfg["eval"]["n_raters"])]

    with open(key, newline="", encoding="utf-8") as f:
        items = [row["item_id"] for row in csv.DictReader(f)]
    found = False
    for sheet in sheets:
        problems = check_sheet(sheet, items)
        if problems:
            found = True
            print(f"{sheet}: {len(problems)} problem(s)")
            for p in problems[:20]:
                print("   ", p)
            if len(problems) > 20:
                print(f"    ... and {len(problems) - 20} more")
    if found and not args.allow_problems:
        sys.exit("Fix the sheets (or rerun with --allow-problems) before scoring.")

    kappas = kappa_per_criterion(sheets[0], sheets[1])
    means = mean_scores_by_system(key, sheets)

    print(f"\n{'Criterion':<26}{'Kappa':>8}")
    for crit in CRITERIA:
        k = kappas[crit]
        print(f"{crit:<26}{'-' if k is None else f'{k:.2f}':>8}")

    print(f"\n{'System':<8}" + "".join(f"{c:>26}" for c in CRITERIA))
    for system in sorted(means):
        print(f"{system:<8}" + "".join(f"{means[system].get(c, float('nan')):>26.2f}" for c in CRITERIA))

    out = ensure_dir(cfg["paths"]["results_dir"]) / "human_eval.json"
    out.write_text(json.dumps({"kappa": kappas, "mean_scores": means}, indent=2))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
