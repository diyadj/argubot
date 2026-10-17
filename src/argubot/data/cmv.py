"""CMV Winning Arguments data (owner: Tricia).

Written and tested: comment cleaning and the thread-level split, because a leak between
train and test would quietly invalidate every result. Left for Tricia: the persona chain
extraction itself, which needs the real ConvoKit corpus open to get right.
"""
from __future__ import annotations

import random
import re
from typing import Dict, Sequence, Tuple

_URL = re.compile(r"https?://\S+|www\.\S+")
_REMOVED = {"", "[deleted]", "[removed]"}
_AUTOMATED_USERS = {"deltabot", "automoderator"}


def is_removed(text: str) -> bool:
    return text.strip().lower() in _REMOVED


def is_automated(speaker: str) -> bool:
    return speaker.strip().lower() in _AUTOMATED_USERS


def clean_comment(text: str) -> str:
    """Drop quoted lines and URLs, collapse whitespace.

    Reddit marks a quote with a leading ">" (sometimes escaped as "&gt;"), so those
    lines go first. Otherwise the bot would be trained to repeat the opponent's own words.
    """
    kept = [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith((">", "&gt;"))
    ]
    text = _URL.sub("", " ".join(kept))
    return " ".join(text.split())


def split_by_thread(
    thread_ids: Sequence[str],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Dict[str, str]:
    """Map each thread id to "train", "val" or "test". A thread never crosses a split.

    Always split threads, not chains. Two chains from one thread share the original post,
    so splitting chains would leak the topic across splits.
    """
    unique = sorted(set(thread_ids))
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    mapping: Dict[str, str] = {}
    for i, tid in enumerate(unique):
        if i < n_train:
            mapping[tid] = "train"
        elif i < n_train + n_val:
            mapping[tid] = "val"
        else:
            mapping[tid] = "test"
    return mapping


def extract_persona_chains(corpus, cfg):
    """TODO (Tricia). The rules from the proposal:

      - A persona is a non-original-poster user who replies to the poster at least twice
        in one thread. Their replies in that thread are the bot's turns, and the poster's
        comments are the user turns.
      - Remove quoted text and URLs (clean_comment), and drop deleted, removed, moderator
        and bot comments (is_removed, is_automated).
      - Keep chains of 3 to 6 turns, with bot replies capped at about 150 tokens.
      - Assign each thread to a split with split_by_thread BEFORE writing any chain out.
      - Write dialogues as JSONL: {"dialogue_id", "split", "turns": [{"speaker", "text"}]}.
        Use the speaker names "user" for the poster and "bot" for the persona.
      - Check the ConvoKit field names against the real corpus. They are easy to get wrong.
    """
    raise NotImplementedError("TODO (Tricia): see the rules in this docstring")
