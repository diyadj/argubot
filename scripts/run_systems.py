"""Run systems A, B, C1, C2 and D on the same scripted debates (owner: Andrea).

Smoke test, needs no models and no data downloads:
    python scripts/run_systems.py --dummy

Real run, once the adapter and the passages file exist:
    python scripts/run_systems.py --dialogues data/processed/test_dialogues.jsonl
"""
import argparse

from argubot.config import ensure_dir, load_config
from argubot.generate.generator import DummyGenerator, HFGenerator
from argubot.io import read_jsonl, write_jsonl
from argubot.nli.scorer import DummyNLIScorer, RobertaMNLIScorer
from argubot.retrieval.bm25 import BM25Retriever
from argubot.retrieval.dummy import DummyRetriever
from argubot.schemas import Dialogue, Passage
from argubot.systems import ArguBotPipeline, build_systems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--systems", nargs="+", default=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--dialogues", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--dummy", action="store_true", help="use fake components and the toy data")
    args = ap.parse_args()

    cfg = load_config(args.config)
    dialogues_path = args.dialogues or ("data/toy/toy_dialogues.jsonl" if args.dummy else cfg["paths"]["test_dialogues"])
    out_dir = ensure_dir(args.out or cfg["paths"]["transcripts_dir"])
    dialogues = [Dialogue.from_dict(d) for d in read_jsonl(dialogues_path)]

    if args.dummy:
        print("DUMMY RUN: fake generator, retriever and NLI. Numbers are meaningless.")
        passages = [Passage.from_dict(d) for d in read_jsonl("data/toy/toy_passages.jsonl")]
        generator, retriever, nli = DummyGenerator(), DummyRetriever(passages), DummyNLIScorer()
    else:
        d = cfg["decoding"]
        generator = HFGenerator(
            cfg["generator"]["base_model"],
            cfg["paths"]["adapter_dir"],
            max_new_tokens=d["max_new_tokens"],
            temperature=d["temperature"],
            top_p=d["top_p"],
            max_prompt_tokens=cfg["training"]["max_seq_len"],
        )
        passages = [Passage.from_dict(x) for x in read_jsonl(cfg["paths"]["passages"])]
        if cfg["retrieval"]["method"] != "bm25":
            raise NotImplementedError("Dense retrieval is Diya's TODO, use method: bm25 for now")
        retriever = BM25Retriever(passages)
        nli = RobertaMNLIScorer(cfg["nli"]["model"], max_length=cfg["nli"]["max_length"])

    systems = build_systems(cfg["decoding"]["n_candidates_rerank"])
    for name in args.systems:
        pipeline = ArguBotPipeline(systems[name], generator, cfg, retriever=retriever, nli=nli)
        transcripts = [pipeline.run_debate(d) for d in dialogues]
        path = f"{out_dir}/{name}.jsonl"
        write_jsonl(path, (t.to_dict() for t in transcripts))
        print(f"System {name}: {len(transcripts)} debates written to {path}")


if __name__ == "__main__":
    main()
