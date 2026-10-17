"""Reranker for system D (owner: Cassey writes it, Andrea plugs it into the pipeline)."""
from __future__ import annotations

from typing import List, Sequence, Tuple

from argubot.nli.scorer import NLIScorer, symmetric_contradiction_batch
from argubot.text import split_sentences


def rerank(
    candidates: Sequence[str],
    claims: Sequence[str],
    scorer: NLIScorer,
    min_words: int = 5,
) -> Tuple[int, List[float]]:
    """Pick the candidate that contradicts the claim log the least.

    Each candidate is scored by its worst sentence-versus-claim contradiction, in both
    directions. Lower is better, and ties go to the earlier candidate.

    Replies shorter than min_words are pushed to the back. Without that, an empty or
    one-word reply would "win" by contradicting nothing, which is the claim-free loophole
    the proposal's guardrail metrics are meant to catch.
    """
    if not candidates:
        raise ValueError("rerank needs at least one candidate")
    if not claims or len(candidates) == 1:
        return 0, [0.0] * len(candidates)

    scores: List[float] = []
    for cand in candidates:
        if len(cand.split()) < min_words:
            scores.append(float("inf"))
            continue
        sentences = split_sentences(cand)
        pairs = [(claim, sent) for claim in claims for sent in sentences]
        worst = max(symmetric_contradiction_batch(scorer, pairs), default=0.0)
        scores.append(worst)

    best = min(range(len(candidates)), key=lambda i: (scores[i], i))
    return best, scores
