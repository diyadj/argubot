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


def build_retriever(cfg: Dict[str, Any], passages) -> Retriever:
    """The retriever named by retrieval.method. Both load a saved index from paths.index_dir if there is one."""
    method = cfg["retrieval"]["method"]
    index_dir = cfg["paths"]["index_dir"]
    if method == "bm25":
        primary: Retriever = BM25Retriever(passages, index_dir=index_dir)
    elif method == "dense":
        from argubot.retrieval.dense import DenseRetriever

        primary = DenseRetriever.load(f"{index_dir}/dense", cfg["retrieval"]["dense_model"], passages)
    else:
        raise ValueError(f"retrieval.method must be bm25 or dense, got {method!r}")
    if not cfg["retrieval"].get("use_backup", False):
        return primary

    from argubot.retrieval.fallback import FallbackRetriever

    backup_passages = [Passage.from_dict(d) for d in read_jsonl(cfg["paths"]["passages_backup"])]
    backup = BM25Retriever(backup_passages, index_dir=f"{index_dir}/ibm_argq")
    return FallbackRetriever(primary, backup, cfg["retrieval"]["backup_min_score"])


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
    retriever = build_retriever(cfg, passages)
    nli = RobertaMNLIScorer(cfg["nli"]["model"], max_length=cfg["nli"]["max_length"])
    return generator, retriever, nli
