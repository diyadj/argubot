"""BM25 keyword baseline (owner: Diya).

This works as written on a small corpus. For all of args.me (about 380k entries, more once
long premises are split) rank_bm25 can be slow per query. If latency hurts, swap in a faster
BM25 implementation behind the same retrieve() method and keep the rest unchanged.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

import numpy as np

from argubot.schemas import Passage
from argubot.text import words


class BM25Retriever:
    def __init__(self, passages: Sequence[Passage]) -> None:
        from rank_bm25 import BM25Okapi

        self.passages = list(passages)
        self._bm25 = BM25Okapi([words(p.text) for p in self.passages])

    def retrieve(self, query: str, k: int = 10) -> List[Passage]:
        scores = self._bm25.get_scores(words(query))
        top = np.argsort(scores)[::-1][:k]
        return [replace(self.passages[i], score=float(scores[i])) for i in top]
