"""Stance-consistency filter for system C2 (owner: Diya, built on Cassey's NLI scorer).

Drops any retrieved passage that contradicts something the bot has already claimed.

The proposal also aligns each args.me stance label to the wording of the debate motion before
filtering. That alignment exists as a tested utility (retrieval/stance_alignment.py) but is not
wired in here, so both runs filter without it (a limitation, README). Passage.stance is carried
through untouched, so the alignment can be added later and the IBM-ArgQ stance test can use it.
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
