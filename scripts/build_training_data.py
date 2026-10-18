"""Build the training and validation examples with claim and evidence slots filled (owner: Tricia and Diya).

    python scripts/build_training_data.py

For every bot turn in a train or val chain, replay the chain with the real claim tracker and
the real retriever, exactly as the pipeline does at inference time, then store:
    {"history": [...], "claims": [...], "passages": [...], "response": "..."}
Training on slots that look different from the inference-time slots would hurt systems C1 to D.
Output: cfg["paths"]["train_examples"] and cfg["paths"]["val_examples"].
The test split is not touched here: scripts/make_test_debates.py picks the scripted test debates from it.

Needs the passages file from scripts/build_indexes.py. Without it, try the smoke test:
    python scripts/build_training_data.py --dummy --limit 50
which uses the toy passages and writes next to the real outputs only if you drop --dummy.
"""
import argparse
import functools
import time
from pathlib import Path

from argubot.config import load_config
from argubot.data.training_data import build_examples, build_examples_parallel
from argubot.io import read_jsonl, write_jsonl
from argubot.retrieval.dummy import DummyRetriever
from argubot.schemas import Passage
from argubot.systems.components import build_retriever


def load_retriever(cfg, passages_path):
    """Runs once in each worker process."""
    return build_retriever(cfg, [Passage.from_dict(d) for d in read_jsonl(passages_path)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--passages", default=None, help="passages JSONL (default: cfg paths.passages)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N dialogues, for a quick check")
    ap.add_argument("--workers", type=int, default=4, help="processes; each holds the passages in memory (~2 GB)")
    ap.add_argument("--dummy", action="store_true", help="toy passages, writes to outputs/dummy/ not data/processed/")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.dummy:
        passages_path = "data/toy/toy_passages.jsonl"
        out_dir = Path(cfg["paths"]["output_dir"]) / "dummy"
        train_out, val_out = out_dir / "train_examples.jsonl", out_dir / "val_examples.jsonl"
    else:
        passages_path = args.passages or cfg["paths"]["passages"]
        train_out, val_out = cfg["paths"]["train_examples"], cfg["paths"]["val_examples"]

    if not Path(passages_path).exists():
        raise SystemExit(
            f"No passages file at {passages_path}. Diya's scripts/build_indexes.py writes it "
            "(args.me needs a manual download). To check the wiring now, add --dummy."
        )
    workers = 1 if args.dummy else max(1, args.workers)

    dialogues = list(read_jsonl(cfg["paths"]["cmv_chains"]))
    if args.limit:
        dialogues = dialogues[: args.limit]
    start = time.time()

    def progress(done: int) -> None:
        if done % 200 == 0:
            rate = done / (time.time() - start)
            print(f"  {done}/{len(dialogues)} dialogues, {rate:.1f}/s, about {(len(dialogues) - done) / rate / 60:.0f} min left", flush=True)

    if workers == 1:
        passages = [Passage.from_dict(d) for d in read_jsonl(passages_path)]
        retriever = DummyRetriever(passages) if args.dummy else build_retriever(cfg, passages)
        print(f"{len(passages)} passages from {passages_path}")
        examples = build_examples(dialogues, retriever, cfg, on_dialogue=progress)
    else:
        print(f"{workers} worker processes, each loading the passages from {passages_path}", flush=True)
        make = functools.partial(load_retriever, cfg, passages_path)
        examples = build_examples_parallel(dialogues, make, cfg, workers=workers, on_dialogue=progress)
    print(f"train: {write_jsonl(str(train_out), examples['train'])} examples -> {train_out}")
    print(f"val:   {write_jsonl(str(val_out), examples['val'])} examples -> {val_out}")


if __name__ == "__main__":
    main()
