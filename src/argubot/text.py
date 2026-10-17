"""Small text helpers shared by the claim tracker, the metrics and the data code."""
from __future__ import annotations

import re
from typing import Iterable, List

# A sentence ends at . ! or ?, optionally followed by a closing quote or bracket. Without the
# second lookbehind, a quoted sentence would swallow the sentence after it.
_SENT_SPLIT = re.compile(
    "(?:(?<=[.!?])|(?<=[.!?][\"'”’)\\]]))\\s+(?=[A-Z0-9\"'“‘(\\[])"
)
_WORD = re.compile(r"[a-z0-9']+")


def flatten_whitespace(text: str) -> str:
    return " ".join(text.split())


def split_sentences(text: str) -> List[str]:
    """Cheap regex sentence splitter. Good enough for short debate replies.

    The claim tracker uses spaCy when it is available, but the metrics use this one
    so that a tracker mistake cannot hide a contradiction from both the system and the metric.
    """
    text = flatten_whitespace(text)
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def words(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def truncate_words(text: str, max_words: int) -> str:
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])
