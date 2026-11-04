"""Build blinded rating sheets for the two human raters (owner: Cassey, logistics: Lumen).

    python scripts/make_rating_sheets.py --config configs/config_4b.yaml
    python scripts/make_rating_sheets.py --config configs/config.yaml --transcripts outputs/transcripts_fix --out outputs/human_eval_fix

Send rater_1.csv and rater_2.csv to the raters. Keep key_PRIVATE.csv away from them
until both have finished, then score them with scripts/score_ratings.py. The whole
procedure is in docs/HUMAN_RATINGS.md.

It never writes over existing sheets, because they may already hold someone's ratings;
pass --overwrite only when you are sure they are blank.
"""
import argparse
import sys
from pathlib import Path

from argubot.config import load_config
from argubot.eval.human_eval import make_rating_sheets
from argubot.io import read_jsonl
from argubot.schemas import Transcript


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--systems", nargs="+", default=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--transcripts", default=None, help="transcripts folder (default: paths.transcripts_dir)")
    ap.add_argument("--out", default=None, help="folder for the sheets (default: paths.rating_dir)")
    ap.add_argument("--overwrite", action="store_true", help="replace sheets that are already there")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out or cfg["paths"]["rating_dir"])
    existing = sorted(p.name for p in out_dir.glob("*.csv")) if out_dir.is_dir() else []
    if existing and not args.overwrite:
        sys.exit(f"{out_dir} already has {existing}. They may hold ratings, so nothing was written. "
                 "Use --out for a new folder, or --overwrite if you are sure they are blank.")
    tdir = Path(args.transcripts or cfg["paths"]["transcripts_dir"])
    print(f"Rating sheets from {tdir} for systems {args.systems}")
    by_system = {
        name: [Transcript.from_dict(d) for d in read_jsonl(str(tdir / f"{name}.jsonl"))]
        for name in args.systems
    }
    paths = make_rating_sheets(
        by_system,
        str(out_dir),
        n_per_system=cfg["eval"]["human_eval_dialogues_per_system"],
        n_raters=cfg["eval"]["n_raters"],
        seed=cfg["seed"],
    )
    for p in paths:
        print("wrote", p)
    print(f"\nGive rater_1.csv to one rater and rater_2.csv to the other. Do not share key_PRIVATE.csv.")


if __name__ == "__main__":
    main()
