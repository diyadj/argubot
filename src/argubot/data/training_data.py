"""Training examples with the claim and evidence slots filled (owner: Tricia and Diya).

Each bot turn of a CMV chain becomes one example. The slots are produced by replaying the
chain the same way ArguBotPipeline.step does at inference time: the tracker has heard the
persona's earlier replies, the query is build_query(latest user turn, claims), and the top
top_k_keep of the top_k_retrieve passages are kept. If this drifted from step(), the adapter
would train on slots that look different from the ones it sees at test time.

Not replayed: the stance filter (systems C2 and D). It needs the NLI model, and the adapter
already sees unfiltered evidence in B and C1, so the filter only ever removes passages from
a slot the model has been trained on.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional

from argubot.claims.tracker import ClaimTracker
from argubot.retrieval.base import Retriever, build_query


def examples_from_dialogue(
    dialogue: Dict[str, Any],
    retriever: Retriever,
    cfg: Dict[str, Any],
    tracker_factory: Optional[Callable[[], ClaimTracker]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield {"history", "claims", "passages", "response"} for every bot turn in one dialogue."""
    r = cfg["retrieval"]
    tracker = (tracker_factory or (
        lambda: ClaimTracker(
            max_claims=cfg["claims"]["max_claims"],
            dedup_threshold=cfg["claims"]["dedup_threshold"],
        )
    ))()
    history = []
    latest_user = None
    for turn in dialogue["turns"]:
        if turn["speaker"] == "user":
            latest_user = turn["text"]
            history.append({"speaker": "user", "text": turn["text"]})
            continue
        if latest_user is None:  # a bot turn with nothing to answer, cannot happen for CMV chains
            continue
        claims = list(tracker.claims)
        query = build_query(latest_user, claims)
        passages = retriever.retrieve(query, k=r["top_k_retrieve"])[: r["top_k_keep"]]
        yield {
            "history": [dict(t) for t in history],
            "claims": claims,
            "passages": [p.text for p in passages],
            "response": turn["text"],
        }
        history.append({"speaker": "bot", "text": turn["text"]})
        tracker.update(turn["text"])
        latest_user = None


def build_examples(
    dialogues: Iterable[Dict[str, Any]],
    retriever: Retriever,
    cfg: Dict[str, Any],
    splits: Iterable[str] = ("train", "val"),
    on_dialogue: Optional[Callable[[int], None]] = None,
) -> Dict[str, list]:
    """Examples grouped by split. Dialogues outside `splits` are skipped (test stays untouched)."""
    wanted = set(splits)
    out: Dict[str, list] = {s: [] for s in wanted}
    for i, d in enumerate(dialogues):
        if d["split"] in wanted:
            out[d["split"]].extend(examples_from_dialogue(d, retriever, cfg))
        if on_dialogue:
            on_dialogue(i + 1)
    return out


# --- parallel version -------------------------------------------------------------------------
# Dialogues are independent, so they can be spread over processes. Each worker loads the retriever
# once (the passages take ~2 GB per worker) and map() hands results back in input order, so the
# output file is identical to the single-process one.

_worker: Dict[str, Any] = {}


def _init_worker(make_retriever: Callable[[], Retriever], cfg: Dict[str, Any]) -> None:
    _worker["retriever"] = make_retriever()
    _worker["cfg"] = cfg


def _work(dialogue: Dict[str, Any]) -> List[Dict[str, Any]]:
    return list(examples_from_dialogue(dialogue, _worker["retriever"], _worker["cfg"]))


def build_examples_parallel(
    dialogues: Iterable[Dict[str, Any]],
    make_retriever: Callable[[], Retriever],
    cfg: Dict[str, Any],
    workers: int = 4,
    splits: Iterable[str] = ("train", "val"),
    on_dialogue: Optional[Callable[[int], None]] = None,
    chunksize: int = 25,
) -> Dict[str, list]:
    """Same result as build_examples. make_retriever must be picklable (a module-level function or a partial)."""
    wanted = set(splits)
    todo = [d for d in dialogues if d["split"] in wanted]
    out: Dict[str, list] = {s: [] for s in wanted}
    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(make_retriever, cfg)) as pool:
        for i, (d, examples) in enumerate(zip(todo, pool.map(_work, todo, chunksize=chunksize))):
            out[d["split"]].extend(examples)
            if on_dialogue:
                on_dialogue(i + 1)
    return out
