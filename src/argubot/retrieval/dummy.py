"""Word-overlap retriever for tests and the --dummy smoke run. Not a baseline, never report it."""
from __future__ import annotations

from dataclasses import replace
from typing import List, Sequence

from argubot.schemas import Passage
from argubot.text import words


class DummyRetriever:
    def __init__(self, passages: Sequence[Passage]) -> None:
        self.passages = list(passages)

    def retrieve(self, query: str, k: int = 10) -> List[Passage]:
        q = set(words(query))
        scored = [
            replace(p, score=float(len(q & set(words(p.text))))) for p in self.passages
        ]
        scored.sort(key=lambda p: p.score, reverse=True)
        return scored[:k]
