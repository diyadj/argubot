"""Fetch the public datasets (owner: Tricia for CMV, Diya for args.me and IBM-ArgQ).

The professor confirmed that a documented download script is fine for large public data,
so keep this script honest: whatever you cannot automate, print exact manual steps for.

    python scripts/download_data.py --cmv
"""
import argparse
from pathlib import Path


def download_cmv(raw_dir: Path) -> None:
    """CMV Winning Arguments through ConvoKit (Tan et al., 2016)."""
    from convokit import Corpus, download

    path = download("winning-args-corpus", data_dir=str(raw_dir / "convokit"))
    corpus = Corpus(filename=path)
    corpus.print_summary_stats()


def explain_manual_steps(raw_dir: Path) -> None:
    print("args.me v1.0-cleaned (TODO Diya: add the exact Zenodo link and file name here)")
    print(f"  save under {raw_dir / 'argsme'}")
    print("IBM-ArgQ-Rank-30k (TODO Diya: add the exact IBM Debater dataset link here)")
    print(f"  save under {raw_dir / 'ibm_argq'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cmv", action="store_true", help="download CMV through ConvoKit")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if args.cmv:
        download_cmv(raw)
    explain_manual_steps(raw)


if __name__ == "__main__":
    main()
