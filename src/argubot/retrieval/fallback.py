"""Backup retrieval (owner: Diya): use the IBM-ArgQ index when the args.me results are weak.

The config says `backup_index: ibm_argq  # used when args.me results are weak`. "Weak" here means the
best args.me passage scores below `backup_min_score`. Only the primary index's own score decides,
because BM25 scores from two different corpora (different idf) cannot be compared with each other.
When it triggers, the answer is the backup index's own top k, ranked by its own score.

Off by default (`retrieval.use_backup: false`), so the training examples built without it stay
consistent with inference. Turn it on and rebuild them together, never one without the other.
"""
from __future__ import annotations

from typing import List

from argubot.retrieval.base import Retriever
from argubot.schemas import Passage


class FallbackRetriever:
    def __init__(self, primary: Retriever, backup: Retriever, min_score: float) -> None:
        self.primary = primary
        self.backup = backup
        self.min_score = min_score
        self.calls = 0
        self.fallbacks = 0

    def retrieve(self, query: str, k: int = 10) -> List[Passage]:
        self.calls += 1
        got = self.primary.retrieve(query, k)
        if got and got[0].score >= self.min_score:
            return got
        self.fallbacks += 1
        return self.backup.retrieve(query, k) or got  # an empty backup must not erase the primary's answer
