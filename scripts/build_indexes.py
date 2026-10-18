"""Build the retrieval passages and indexes (owner: Diya).

    python scripts/build_indexes.py

Reads data/raw/argsme and data/raw/ibm_argq (see scripts/download_data.py) and writes:
  cfg["paths"]["passages"]         args.me, one passage per line with stance and motion (main index)
  cfg["paths"]["passages_backup"]  IBM-ArgQ in the same format (backup index and the 71-motion stance test)
  cfg["paths"]["index_dir"]        the BM25 index for the main passages, so nobody rebuilds it per run
The two sources stay in separate files so the short, crowd-written IBM arguments cannot crowd
the args.me results out of the main index.

With --dense it also encodes every args.me passage with MiniLM into <index_dir>/dense. That takes hours
on a CPU and minutes on a GPU, so run it on Colab and copy the folder back. Set retrieval.method to
"dense" in the config to use it; scripts load the saved index and never re-encode.
"""
import argparse
import time
from collections import Counter
from pathlib import Path

from argubot.config import load_config
from argubot.data.argsme import load_argsme, load_ibm_argq
from argubot.io import write_jsonl
from argubot.retrieval.bm25 import BM25Retriever
from dataclasses import asdict as asdict_passage


def describe(name: str, passages) -> None:
    stance = Counter(p.stance for p in passages)
    motions = len({p.motion for p in passages})
    print(f"{name}: {len(passages)} passages, {motions} distinct motions, stances {dict(stance)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--skip-index", action="store_true", help="write the passage files only")
    ap.add_argument("--dense", action="store_true", help="also build the MiniLM + FAISS index (slow without a GPU)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    raw = Path(cfg["paths"]["raw_dir"])

    start = time.time()
    argsme = load_argsme(str(raw / "argsme"), cfg["data"]["passage_max_words"])
    describe("args.me", argsme)
    write_jsonl(cfg["paths"]["passages"], (asdict_passage(p) for p in argsme))

    ibm = load_ibm_argq(str(raw / "ibm_argq"))
    describe("IBM-ArgQ", ibm)
    write_jsonl(cfg["paths"]["passages_backup"], (asdict_passage(p) for p in ibm))
    print(f"passage files written in {time.time() - start:.0f}s")

    if not args.skip_index:
        start = time.time()
        BM25Retriever(argsme, index_dir=cfg["paths"]["index_dir"])
        print(f"BM25 index for args.me saved to {cfg['paths']['index_dir']} in {time.time() - start:.0f}s")
        BM25Retriever(ibm, index_dir=f"{cfg['paths']['index_dir']}/ibm_argq")
        print(f"BM25 index for IBM-ArgQ (the backup) saved to {cfg['paths']['index_dir']}/ibm_argq")

    if args.dense:
        from argubot.retrieval.dense import DenseRetriever

        start = time.time()
        dense = DenseRetriever(cfg["retrieval"]["dense_model"])
        dense.build(
            argsme,
            on_progress=lambda done, total: print(f"  encoded {done}/{total} passages, {time.time() - start:.0f}s", flush=True),
        )
        dense.save(f"{cfg['paths']['index_dir']}/dense")
        print(f"dense index saved to {cfg['paths']['index_dir']}/dense in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
