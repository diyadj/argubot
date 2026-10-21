"""NLI scoring (owner: Cassey).

One small interface, three jobs: the stance filter, the reranker and the contradiction-rate
metric all call it. RoBERTa-large-MNLI is the real scorer. DummyNLIScorer exists so everyone
can run and test their own lane before the real model is wired up. Never report numbers from it.
"""
from __future__ import annotations

import re
from typing import List, Optional, Protocol, Sequence, Tuple

from argubot.schemas import NLIScore


class NLIScorer(Protocol):
    def score(self, premise: str, hypothesis: str) -> NLIScore: ...

    def score_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIScore]: ...


class RobertaMNLIScorer:
    """roberta-large-mnli. Labels are read from the model config instead of assumed."""

    def __init__(
        self,
        model_name: str = "roberta-large-mnli",
        device: Optional[str] = None,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self._idx = {label.lower(): i for i, label in self.model.config.id2label.items()}
        for needed in ("contradiction", "neutral", "entailment"):
            if needed not in self._idx:
                raise ValueError(f"Model labels {self._idx} are missing '{needed}'")

    def score(self, premise: str, hypothesis: str) -> NLIScore:
        return self.score_batch([(premise, hypothesis)])[0]

    def score_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIScore]:
        torch = self.torch
        out: List[NLIScore] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start : start + self.batch_size]
            enc = self.tokenizer(
                [p for p, _ in chunk],
                [h for _, h in chunk],
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                probs = torch.softmax(self.model(**enc).logits, dim=-1).cpu().tolist()
            for row in probs:
                out.append(
                    NLIScore(
                        contradiction=row[self._idx["contradiction"]],
                        neutral=row[self._idx["neutral"]],
                        entailment=row[self._idx["entailment"]],
                    )
                )
        return out


_NEGATIONS = {
    "not", "no", "never", "cannot", "can't", "isn't", "aren't", "doesn't", "don't",
    "won't", "wasn't", "none", "nobody", "nothing", "neither",
}


class DummyNLIScorer:
    """Word-overlap and negation heuristic. Only for tests and the --dummy smoke run."""

    def score(self, premise: str, hypothesis: str) -> NLIScore:
        p = set(re.findall(r"[a-z']+", premise.lower()))
        h = set(re.findall(r"[a-z']+", hypothesis.lower()))
        if not p or not h:
            return NLIScore(0.1, 0.8, 0.1)
        overlap = len(p & h) / min(len(p), len(h))
        negation_differs = bool(p & _NEGATIONS) != bool(h & _NEGATIONS)
        if overlap >= 0.5 and negation_differs:
            return NLIScore(0.9, 0.05, 0.05)
        if overlap >= 0.7:
            return NLIScore(0.05, 0.15, 0.8)
        return NLIScore(0.1, 0.8, 0.1)

    def score_batch(self, pairs: Sequence[Tuple[str, str]]) -> List[NLIScore]:
        return [self.score(p, h) for p, h in pairs]


def symmetric_contradiction_batch(scorer: NLIScorer, pairs: Sequence[Tuple[str, str]]) -> List[float]:
    """Contradiction probability for each pair, taking the larger of both directions.

    NLI is not symmetric, so A contradicting B can score differently from B contradicting A.
    The proposal checks both directions, and so does everything built on this function.
    """
    if not pairs:
        return []
    forward = list(pairs)
    backward = [(h, p) for p, h in pairs]
    scores = scorer.score_batch(forward + backward)
    n = len(pairs)
    return [max(scores[i].contradiction, scores[n + i].contradiction) for i in range(n)]
