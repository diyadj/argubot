"""Build blinded rating sheets for the two human raters (owner: Cassey, logistics: Lumen).

    python scripts/make_rating_sheets.py

Send rater_1.csv and rater_2.csv to the raters. Keep key_PRIVATE.csv away from them
until both have finished. Afterwards, use kappa_per_criterion and mean_scores_by_system
from argubot.eval.human_eval to turn the filled-in sheets into numbers.
"""
import argparse
from pathlib import Path

from argubot.config import load_config
from argubot.eval.human_eval import make_rating_sheets
from argubot.io import read_jsonl
from argubot.schemas import Transcript


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--systems", nargs="+", default=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--transcripts", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    tdir = Path(args.transcripts or cfg["paths"]["transcripts_dir"])
    by_system = {
        name: [Transcript.from_dict(d) for d in read_jsonl(str(tdir / f"{name}.jsonl"))]
        for name in args.systems
    }
    paths = make_rating_sheets(
        by_system,
        cfg["paths"]["rating_dir"],
        n_per_system=cfg["eval"]["human_eval_dialogues_per_system"],
        n_raters=cfg["eval"]["n_raters"],
        seed=cfg["seed"],
    )
    for p in paths:
        print("wrote", p)


if __name__ == "__main__":
    main()
