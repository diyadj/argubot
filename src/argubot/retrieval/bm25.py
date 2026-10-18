"""BM25 keyword retriever (owner: Diya).

An inverted index in numpy. rank_bm25 scores a query by looping over every document in Python,
which took about 19 seconds per query on the 1.18M args.me passages (about 100 hours for the
training set). Here each term keeps one array of the documents containing it and one array of
that term's finished BM25 weight in each, so a query only touches the documents that share a
word with it.

The scoring is Okapi BM25 exactly as rank_bm25 computes it (k1=1.5, b=0.75, epsilon=0.25 for
negative idf), and tests/test_bm25.py checks the two against each other.
"""
from __future__ import annotations

import json
from array import array
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from argubot.schemas import Passage
from argubot.text import words

_FILES = ("bm25_docs.npy", "bm25_weights.npy", "bm25_offsets.npy", "bm25_vocab.json")


class BM25Retriever:
    def __init__(
        self,
        passages: Sequence[Passage],
        index_dir: Optional[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
    ) -> None:
        """Pass index_dir to reuse a saved index (built and saved there the first time)."""
        self.passages = list(passages)
        if index_dir and self._load(Path(index_dir)):
            return
        self._build(k1, b, epsilon)
        if index_dir:
            self.save(index_dir)

    def _build(self, k1: float, b: float, epsilon: float) -> None:
        n_docs = len(self.passages)
        vocab: Dict[str, int] = {}
        doc_of, term_of, freq_of = array("i"), array("i"), array("i")
        lengths = np.zeros(n_docs, dtype=np.float64)
        for d, passage in enumerate(self.passages):
            tokens = words(passage.text)
            lengths[d] = len(tokens)
            for term, f in Counter(tokens).items():
                doc_of.append(d)
                term_of.append(vocab.setdefault(term, len(vocab)))
                freq_of.append(f)
        doc_ids = np.frombuffer(doc_of, dtype=np.int32)
        term_ids = np.frombuffer(term_of, dtype=np.int32)
        freqs = np.frombuffer(freq_of, dtype=np.int32).astype(np.float64)

        # idf the way rank_bm25 does it: negative values are replaced by epsilon * the average idf
        n_vocab = len(vocab)
        df = np.bincount(term_ids, minlength=n_vocab).astype(np.float64)
        idf = np.log(n_docs - df + 0.5) - np.log(df + 0.5)
        if n_vocab:
            idf[idf < 0] = epsilon * idf.mean()

        avgdl = lengths.mean() if n_docs else 1.0
        norm = k1 * (1 - b + b * lengths[doc_ids] / avgdl)
        weights = (idf[term_ids] * freqs * (k1 + 1) / (freqs + norm)).astype(np.float32)

        order = np.argsort(term_ids, kind="stable")
        self._docs = doc_ids[order]
        self._weights = weights[order]
        self._offsets = np.concatenate([[0], np.cumsum(df.astype(np.int64))])
        self._vocab = vocab

    def save(self, index_dir: str) -> None:
        d = Path(index_dir)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / _FILES[0], self._docs)
        np.save(d / _FILES[1], self._weights)
        np.save(d / _FILES[2], self._offsets)
        (d / _FILES[3]).write_text(
            json.dumps({"n_docs": len(self.passages), "vocab": self._vocab}), encoding="utf-8"
        )

    def _load(self, d: Path) -> bool:
        if not all((d / name).exists() for name in _FILES):
            return False
        meta = json.loads((d / _FILES[3]).read_text(encoding="utf-8"))
        if meta["n_docs"] != len(self.passages):  # the passages file changed, so the cache is stale
            return False
        self._docs = np.load(d / _FILES[0])
        self._weights = np.load(d / _FILES[1])
        self._offsets = np.load(d / _FILES[2])
        self._vocab = meta["vocab"]
        return True

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.passages), dtype=np.float32)
        # A repeated query word counts again, as in rank_bm25, so walk each distinct word once and
        # multiply. Queries repeat words like "the" dozens of times and those have the longest lists.
        for term, count in Counter(words(query)).items():
            t = self._vocab.get(term)
            if t is not None:
                lo, hi = self._offsets[t], self._offsets[t + 1]
                out[self._docs[lo:hi]] += self._weights[lo:hi] * count if count > 1 else self._weights[lo:hi]
        return out

    def retrieve(self, query: str, k: int = 10) -> List[Passage]:
        scores = self.scores(query)
        k = min(k, len(scores))
        if k == 0:
            return []
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]
        return [replace(self.passages[i], score=float(scores[i])) for i in top]
