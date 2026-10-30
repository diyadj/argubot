"""Claim tracker (owner: Tricia).

Pulls declarative sentences out of the bot's own replies and keeps the most recent few,
de-duplicated. This is the log that conditions the generator in systems C1, C2 and D.

What works now: a regex fallback that runs anywhere, plus a spaCy path when
en_core_web_sm is installed. What Tricia should improve is listed in the TODOs below.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional, Tuple

from argubot.text import is_finished, jaccard, split_sentences, trim_unfinished, words

MIN_WORDS = 4
MAX_CLAIM_WORDS = 40  # longer than this is a run-on, which makes a poor claim for the NLI check
_HEDGE = re.compile(r"\b(maybe|perhaps|might|possibly|could be|not sure)\b", re.IGNORECASE)

# A concession gives ground before the speaker's real point: "Uniforms can be cheaper, but ...".
# The part before "but" is not what the bot is arguing, so only the part after it is stored.
_BUT = re.compile(r",?\s+but,?\s+", re.IGNORECASE)
_CONCEDES = re.compile(
    r"^(?:sure|granted|true|admittedly|of course|yes|okay|ok|fair enough)\b|\b(?:can|may|could|might)\b(?!')",
    re.IGNORECASE,
)
_BECAUSE = re.compile(r",?\s+because\b", re.IGNORECASE)

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
    if s.endswith("!") and len(s.split()) < 10:  # "OK, now we're getting somewhere!"
        return False
    if s.startswith((">", '"', "“")):  # quoting the opponent, not the bot's own claim
        return False
    return not _HEDGE.search(s)


def _end(text: str) -> str:
    text = text.strip()
    text = text[0].upper() + text[1:] if text else text
    return text if text.endswith((".", "!", "?")) else text + "."


def _refine(sentence: str) -> List[str]:
    """Turn one sentence into the claim(s) it makes: drop a conceded half, split on ";", cut run-ons."""
    s = sentence.strip()
    m = _BUT.search(s)
    if m and _CONCEDES.search(s[: m.start()]) and len(s[m.end():].split()) >= MIN_WORDS:
        s = s[m.end():]
    out = []
    for part in (p for p in s.split(";") if p.strip()):
        part = _end(part)
        if len(part.split()) > MAX_CLAIM_WORDS:
            b = _BECAUSE.search(part)
            part = _end(part[: b.start()]) if b else part
        if len(part.split()) <= MAX_CLAIM_WORDS:
            out.append(part)
    return out


def extract_claims(text: str, use_spacy: bool = True) -> List[str]:
    """Return the declarative, un-hedged claims in a reply.

    Tuned on 50 real CMV bot replies. What the filters do, and why:
      - questions, short exclamations, quotes and hedged sentences ("maybe", "might") are skipped,
        and so is an unfinished last sentence (generation hit max_new_tokens).
        "I think X" is kept: in a debate it is still the speaker's position.
      - a concession ("uniforms can be cheaper, but ...") keeps only the part after "but".
      - a sentence with ";" becomes one claim per clause, and a sentence over MAX_CLAIM_WORDS
        is cut before "because" or dropped, since a run-on is a poor claim to check for contradictions.
    """
    # Drop an unfinished ending first: spaCy can split one cut-off sentence into two "sentences".
    text = trim_unfinished(text)
    if not is_finished(text):
        return []  # not one complete sentence
    nlp = _load_spacy() if use_spacy else None
    if nlp is None:
        sentences = split_sentences(text)
        ok = lambda _s: True
    else:
        sentences = [sent for sent in nlp(text).sents]
        ok = lambda sent: any(t.dep_ in ("nsubj", "nsubjpass", "expl") for t in sent) and any(
            t.pos_ in ("VERB", "AUX") for t in sent
        )

    claims: List[str] = []
    for sent in sentences:
        raw = sent if isinstance(sent, str) else sent.text
        if not ok(sent):
            continue
        claims.extend(c for c in _refine(raw) if _is_candidate(c))
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
