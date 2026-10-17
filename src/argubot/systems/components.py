"""Builds the generator, retriever and NLI scorer from the config.

run_systems.py and the demo server both call this, so a real run and the live demo
always load the same things the same way.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from argubot.generate.generator import DummyGenerator, Generator, HFGenerator
from argubot.io import read_jsonl
from argubot.nli.scorer import DummyNLIScorer, NLIScorer, RobertaMNLIScorer
from argubot.retrieval.base import Retriever
from argubot.retrieval.bm25 import BM25Retriever
from argubot.retrieval.dummy import DummyRetriever
from argubot.schemas import Passage


def load_components(cfg: Dict[str, Any], dummy: bool = False) -> Tuple[Generator, Retriever, NLIScorer]:
    if dummy:
        passages = [Passage.from_dict(d) for d in read_jsonl("data/toy/toy_passages.jsonl")]
        return DummyGenerator(), DummyRetriever(passages), DummyNLIScorer()

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
    return generator, retriever, nli
