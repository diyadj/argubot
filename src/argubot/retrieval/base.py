"""Retriever interface (owner: Diya). The pipeline only ever calls retrieve()."""
from __future__ import annotations

from typing import List, Protocol, Sequence

from argubot.schemas import Passage


class Retriever(Protocol):
    def retrieve(self, query: str, k: int = 10) -> List[Passage]:
        """Return up to k passages, best first, with Passage.score filled in."""
        ...


def build_query(user_text: str, claims: Sequence[str]) -> str:
    """The proposal's query: the user's latest message plus the bot's claim log.

    Systems A and B pass an empty claim list, so for them the query is the user message alone.
    That keeps the claim log from leaking into the baseline.
    """
    return " ".join([user_text.strip(), *[c.strip() for c in claims]]).strip()
