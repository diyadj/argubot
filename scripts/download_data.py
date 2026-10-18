"""Fetch the public datasets (owner: Tricia for CMV, Diya for args.me and IBM-ArgQ).

    python scripts/download_data.py --all          # everything, about 250 MB
    python scripts/download_data.py --cmv          # CMV Winning Arguments via ConvoKit
    python scripts/download_data.py --argsme       # args.me v1.0-cleaned from Zenodo (237 MB)
    python scripts/download_data.py --ibm          # IBM-ArgQ-Rank-30k from Hugging Face (5 MB)

Files that already exist are skipped, so rerunning is safe. Zenodo can be slow, so allow a few minutes.
"""
import argparse
import shutil
import urllib.request
from pathlib import Path

# args.me corpus v1.0-cleaned (Ajjour et al., 2019): 382,545 arguments, one zip holding args-me-1.0-cleaned.json.
ARGSME_URL = "https://zenodo.org/api/records/4139439/files/argsme-1.0-cleaned.zip/content"
ARGSME_FILE = "argsme-1.0-cleaned.zip"
ARGSME_BYTES = 236_787_622

# IBM-ArgQ-Rank-30k (Gretz et al., 2020): 30,497 arguments on 71 topics. IBM's own download page has been
# taken down (404), but IBM publishes the same files on Hugging Face as ibm-research/argument_quality_ranking_30k.
IBM_BASE = "https://huggingface.co/datasets/ibm-research/argument_quality_ranking_30k/resolve/main/"
IBM_FILES = ("train.csv", "dev.csv", "test.csv")


def download_cmv(raw_dir: Path) -> None:
    """CMV Winning Arguments through ConvoKit (Tan et al., 2016)."""
    from convokit import Corpus, download

    path = download("winning-args-corpus", data_dir=str(raw_dir / "convokit"))
    corpus = Corpus(filename=path)
    corpus.print_summary_stats()


def fetch(url: str, dest: Path, expected_bytes: int = 0) -> None:
    if dest.exists() and (not expected_bytes or dest.stat().st_size == expected_bytes):
        print(f"already have {dest}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n  -> {dest}")
    with urllib.request.urlopen(url, timeout=60) as response, open(tmp, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if expected_bytes and tmp.stat().st_size != expected_bytes:
        raise SystemExit(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {expected_bytes}. Try again.")
    tmp.replace(dest)


def download_argsme(raw_dir: Path) -> None:
    fetch(ARGSME_URL, raw_dir / "argsme" / ARGSME_FILE, ARGSME_BYTES)


def download_ibm_argq(raw_dir: Path) -> None:
    for name in IBM_FILES:
        fetch(IBM_BASE + name, raw_dir / "ibm_argq" / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--cmv", action="store_true", help="download CMV through ConvoKit")
    ap.add_argument("--argsme", action="store_true", help="download args.me v1.0-cleaned from Zenodo")
    ap.add_argument("--ibm", action="store_true", help="download IBM-ArgQ-Rank-30k from Hugging Face")
    ap.add_argument("--all", action="store_true", help="all three")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    if not (args.cmv or args.argsme or args.ibm or args.all):
        ap.print_help()
        return
    if args.cmv or args.all:
        download_cmv(raw)
    if args.argsme or args.all:
        download_argsme(raw)
    if args.ibm or args.all:
        download_ibm_argq(raw)


if __name__ == "__main__":
    main()
