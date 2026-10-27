"""Run systems A, B, C1, C2 and D on the same scripted debates (owner: Andrea).

Smoke test, needs no models and no data downloads:
    python scripts/run_systems.py --dummy

Real run, once the adapter and the passages file exist:
    python scripts/run_systems.py --dialogues data/processed/test_dialogues.jsonl

Quick check on the first 3 debates, kept apart from the real transcripts:
    python scripts/run_systems.py --limit 3 --out outputs/check3

On the cluster, scripts/violet_run.sbatch runs this as a batch job, one system at a time.
"""
import argparse

from argubot.config import ensure_dir, load_config
from argubot.io import read_jsonl, write_jsonl
from argubot.schemas import Dialogue
from argubot.systems import ArguBotPipeline, build_systems
from argubot.systems.components import load_components


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--systems", nargs="+", default=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--dialogues", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only the first N debates")
    ap.add_argument("--dummy", action="store_true", help="use fake components and the toy data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dialogues_path = args.dialogues or ("data/toy/toy_dialogues.jsonl" if args.dummy else cfg["paths"]["test_dialogues"])
    out_dir = ensure_dir(args.out or cfg["paths"]["transcripts_dir"])
    dialogues = [Dialogue.from_dict(d) for d in read_jsonl(dialogues_path)][: args.limit]

    if args.dummy:
        print("DUMMY RUN: fake generator, retriever and NLI. Numbers are meaningless.")
    generator, retriever, nli = load_components(cfg, dummy=args.dummy)

    systems = build_systems(cfg["decoding"]["n_candidates_rerank"])
    for name in args.systems:
        pipeline = ArguBotPipeline(systems[name], generator, cfg, retriever=retriever, nli=nli)
        transcripts = [pipeline.run_debate(d) for d in dialogues]
        path = f"{out_dir}/{name}.jsonl"
        write_jsonl(path, (t.to_dict() for t in transcripts))
        print(f"System {name}: {len(transcripts)} debates written to {path}")


if __name__ == "__main__":
    main()
