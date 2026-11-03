"""Check that a stage has everything it needs before spending a GPU job on it.

    python scripts/preflight.py --config configs/config_4b.yaml --stage timing
    python scripts/preflight.py --config configs/config_4b.yaml --stage train
    python scripts/preflight.py --config configs/config_4b.yaml --stage run
    python scripts/preflight.py --config configs/config_4b.yaml --stage eval
    python scripts/preflight.py --config configs/config_4b.yaml --stage demo

Prints one line per check (OK, WARN, FAIL, SKIP) and exits with 1 if anything FAILs.
The batch jobs run it first, so you only need it by hand to check things on the login node.
"""
import argparse
import sys

from argubot.config import load_config
from argubot.preflight import STAGES, check_stage, failed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--overwrite", action="store_true", help="train: an existing adapter will be replaced")
    ap.add_argument("--limit", type=int, default=None, help="run/eval: only the first N debates")
    ap.add_argument("--transcripts", default=None, help="run/eval: transcripts folder (default from the config)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(f"preflight for stage '{args.stage}' with {args.config}")
    checks = check_stage(cfg, args.stage, overwrite=args.overwrite, limit=args.limit,
                         transcripts_dir=args.transcripts)
    for c in checks:
        print(c.line())
    if failed(checks):
        print("\nSomething FAILED above. Fix it before submitting the job.")
        sys.exit(1)
    print("\nAll set.")


if __name__ == "__main__":
    main()
