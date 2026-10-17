"""Claim tracker (owner: Tricia).

Pulls declarative sentences out of the bot's own replies and keeps the most recent few,
de-duplicated. This is the log that conditions the generator in systems C1, C2 and D.

What works now: a regex fallback that runs anywhere, plus a spaCy path when
en_core_web_sm is installed. What Tricia should improve is listed in the TODOs below.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from argubot.text import jaccard, split_sentences, words

MIN_WORDS = 4
_HEDGE = re.compile(r"\b(maybe|perhaps|might|possibly|could be|not sure)\b", re.IGNORECASE)

_nlp = None
_nlp_failed = False


def _load_spacy():
    """Load spaCy once. Returns None when it or the model is missing, so the regex path takes over."""
    global _nlp, _nlp_failed
    if _nlp is None and not _nlp_failed:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            _nlp_failed = True
    return _nlp


def _is_candidate(sentence: str) -> bool:
    s = sentence.strip()
    if len(s.split()) < MIN_WORDS:
        return False
    if s.endswith("?"):
        return False
    if s.startswith((">", '"', "“")):  # quoting the opponent, not the bot's own claim
        return False
    return not _HEDGE.search(s)


def extract_claims(text: str, use_spacy: bool = True) -> List[str]:
    """Return the declarative, un-hedged sentences in a reply.

    TODO (Tricia): check this on 50 real CMV replies and tune. Things to look at:
      - "I think X" is still a claim in a debate, so hedges like that should stay in.
      - Concessions ("uniforms can be cheaper, but ...") should not be stored as claims.
      - Very long sentences may need to be split on "because" or semicolons.
    """
    nlp = _load_spacy() if use_spacy else None
    if nlp is None:
        return [s for s in split_sentences(text) if _is_candidate(s)]

    claims: List[str] = []
    for sent in nlp(text).sents:
        if not _is_candidate(sent.text):
            continue
        has_subject = any(t.dep_ in ("nsubj", "nsubjpass", "expl") for t in sent)
        has_verb = any(t.pos_ in ("VERB", "AUX") for t in sent)
        if has_subject and has_verb:
            claims.append(sent.text.strip())
    return claims


class ClaimTracker:
    """Keeps the max_claims most recent distinct claims, oldest first."""

    def __init__(
        self,
        max_claims: int = 5,
        dedup_threshold: float = 0.8,
        extractor: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.max_claims = max_claims
        self.dedup_threshold = dedup_threshold
        self._extract = extractor or extract_claims
        self._items: List[Tuple[str, List[str]]] = []

    @property
    def claims(self) -> List[str]:
        return [claim for claim, _ in self._items]

    def update(self, bot_reply: str) -> List[str]:
        """Add claims from a new bot reply. Returns only the claims that were new."""
        added: List[str] = []
        for claim in self._extract(bot_reply):
            key = words(claim)
            dup = next(
                (i for i, (_, k) in enumerate(self._items) if jaccard(k, key) >= self.dedup_threshold),
                None,
            )
            if dup is not None:
                # The bot said it again, so refresh its recency instead of storing it twice.
                self._items.append(self._items.pop(dup))
            else:
                self._items.append((claim, key))
                added.append(claim)
        self._items = self._items[-self.max_claims :]
        return added

    def reset(self) -> None:
        self._items = []
