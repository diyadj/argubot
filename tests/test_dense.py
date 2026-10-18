import numpy as np
import pytest

from argubot.retrieval.dense import DenseRetriever
from argubot.schemas import Passage
from argubot.text import words

DIM = 128


def fake_encoder(texts):
    """Bag of hashed words, unit length. Crude, but passages that share words land close together."""
    out = np.zeros((len(texts), DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        for w in words(text):
            out[row, sum(map(ord, w)) % DIM] += 1.0
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


PASSAGES = [
    Passage("School uniforms reduce bullying and clothing differences.", "PRO", motion="uniforms"),
    Passage("The death penalty does not deter violent crime.", "CON", motion="death penalty"),
    Passage("Free condoms in schools lower teen pregnancy.", "PRO", motion="condoms"),
    Passage("Uniforms cost families money every single year.", "CON", motion="uniforms"),
]


def built(**kwargs):
    r = DenseRetriever(model_name="fake", encoder=fake_encoder)
    r.build(PASSAGES, **kwargs)
    return r


def test_nearest_passage_is_the_one_that_shares_the_most_words():
    got = built().retrieve("do uniforms reduce bullying", k=2)
    assert got[0].text.startswith("School uniforms reduce bullying")
    assert [p.score for p in got] == sorted((p.score for p in got), reverse=True)
    assert got[0].stance == "PRO" and got[0].motion == "uniforms"  # metadata survives retrieval


def test_chunked_build_gives_the_same_results_and_reports_progress():
    seen = []
    chunked = built(chunk=3, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(3, 4), (4, 4)]
    assert [p.text for p in chunked.retrieve("death penalty crime", k=4)] == [
        p.text for p in built().retrieve("death penalty crime", k=4)
    ]


def test_k_larger_than_the_corpus_and_unbuilt_retriever():
    assert len(built().retrieve("uniforms", k=50)) == len(PASSAGES)
    assert DenseRetriever(encoder=fake_encoder).retrieve("uniforms") == []


def test_save_and_load_round_trip_and_stale_index_is_refused(tmp_path):
    built().save(str(tmp_path))
    loaded = DenseRetriever.load(str(tmp_path), "fake", PASSAGES, encoder=fake_encoder)
    assert [p.text for p in loaded.retrieve("condoms teen pregnancy", k=1)] == [PASSAGES[2].text]
    with pytest.raises(ValueError, match="passages"):
        DenseRetriever.load(str(tmp_path), "fake", PASSAGES[:2], encoder=fake_encoder)
    with pytest.raises(ValueError, match="built with"):
        DenseRetriever.load(str(tmp_path), "other-model", PASSAGES, encoder=fake_encoder)
