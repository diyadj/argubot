"""Build the retrieval passages and indexes (owner: Diya).

    python scripts/build_indexes.py

Output: cfg["paths"]["passages"] (one passage per line, with stance), then the BM25 and
FAISS indexes under cfg["paths"]["index_dir"]. Build the IBM-ArgQ backup index the same way.
"""
import argparse

from argubot.config import load_config
from argubot.data.argsme import load_argsme


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    # TODO (Diya): 1) load_argsme and load_ibm_argq, write the shared passages JSONL.
    #              2) build and save the dense index with DenseRetriever.build and .save.
    #              3) report passage counts so the report's dataset table is exact.
    load_argsme(cfg["paths"]["raw_dir"] + "/argsme")


if __name__ == "__main__":
    main()
