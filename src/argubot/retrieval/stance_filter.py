"""Stance-consistency filter for system C2 (owner: Diya, built on Cassey's NLI scorer).

Drops any retrieved passage that contradicts something the bot has already claimed.

TODO (Diya): the proposal says each args.me stance label is aligned to the wording of the
debate motion before filtering. That alignment is not done here. Passage.stance is carried
through untouched so you can add it, and so the IBM-ArgQ stance test can use it.
"""
from __future__ import annotations

from typing import List, Sequence

from argubot.nli.scorer import NLIScorer, symmetric_contradiction_batch
from argubot.schemas import Passage


class StanceConsistencyFilter:
    def __init__(self, scorer: NLIScorer, threshold: float = 0.5) -> None:
        self.scorer = scorer
        self.threshold = threshold

    def filter(self, passages: Sequence[Passage], claims: Sequence[str]) -> List[Passage]:
        if not claims:
            return list(passages)
        kept: List[Passage] = []
        for passage in passages:
            pairs = [(claim, passage.text) for claim in claims]
            worst = max(symmetric_contradiction_batch(self.scorer, pairs), default=0.0)
            if worst < self.threshold:
                kept.append(passage)
        return kept
