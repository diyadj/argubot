import numpy as np
from rank_bm25 import BM25Okapi

from argubot.io import read_jsonl
from argubot.retrieval.bm25 import BM25Retriever
from argubot.schemas import Passage
from argubot.text import words

CORPUS = [
    "School uniforms reduce bullying because clothing differences matter less.",
    "Uniforms limit self expression which matters for students.",
    "The death penalty does not deter crime according to most studies.",
    "Free condoms in high schools reduce teen pregnancy rates.",
    "Uniforms save families money every year on clothes.",
    "The the the common word common word appears here.",
    "A short one.",
]


def passages():
    return [Passage(text=t, source_id=str(i)) for i, t in enumerate(CORPUS)]


def test_scores_match_rank_bm25():
    ours = BM25Retriever(passages())
    reference = BM25Okapi([words(t) for t in CORPUS])
    for query in ("uniforms bullying", "the common word", "death penalty uniforms uniforms", "nothing matches xyzzy"):
        assert np.allclose(ours.scores(query), reference.get_scores(words(query)), rtol=1e-4, atol=1e-5), query


def test_retrieve_returns_best_first_with_scores():
    got = BM25Retriever(passages()).retrieve("uniforms bullying", k=3)
    assert got[0].text.startswith("School uniforms reduce bullying")
    assert [p.score for p in got] == sorted((p.score for p in got), reverse=True)
    assert len(got) == 3


def test_k_larger_than_the_corpus_and_empty_corpus_are_fine():
    assert len(BM25Retriever(passages()).retrieve("uniforms", k=100)) == len(CORPUS)
    assert BM25Retriever([]).retrieve("uniforms") == []


def test_saved_index_gives_identical_results_and_goes_stale_with_the_passages(tmp_path):
    first = BM25Retriever(passages(), index_dir=str(tmp_path))
    again = BM25Retriever(passages(), index_dir=str(tmp_path))  # loads the saved files
    assert [p.text for p in first.retrieve("uniforms money")] == [p.text for p in again.retrieve("uniforms money")]
    fewer = BM25Retriever(passages()[:3], index_dir=str(tmp_path))  # different size, so rebuilt
    assert len(fewer.passages) == 3 and len(fewer.retrieve("uniforms", k=10)) == 3


def test_toy_passages_still_retrieve_the_obvious_match():
    toy = [Passage.from_dict(d) for d in read_jsonl("data/toy/toy_passages.jsonl")]
    best = BM25Retriever(toy).retrieve("uniforms save families money", k=1)[0]
    assert "money" in best.text
