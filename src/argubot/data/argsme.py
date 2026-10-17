"""args.me and IBM-ArgQ loading (owner: Diya).

Written and tested: splitting a long premise into passages that fit the NLI model.
Left for Diya: reading the real files, because the field names must be checked against
the download itself.
"""
from __future__ import annotations

from typing import List

from argubot.text import split_sentences


def split_into_passages(text: str, max_words: int = 120) -> List[str]:
    """Pack whole sentences into passages of at most max_words words.

    RoBERTa-MNLI reads at most 512 tokens, so a premise of several paragraphs has to be
    cut up first. A single sentence longer than max_words becomes its own passage and the
    scorer truncates it.
    """
    passages: List[str] = []
    current: List[str] = []
    count = 0
    for sentence in split_sentences(text):
        n = len(sentence.split())
        if current and count + n > max_words:
            passages.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        passages.append(" ".join(current))
    return passages


def load_argsme(path: str):
    """TODO (Diya). Each args.me entry has a conclusion, premises with a PRO or CON stance,
    and some context. Turn each premise into one or more Passage objects with
    split_into_passages, keep the stance and the conclusion (as Passage.motion), and
    write them to cfg["paths"]["passages"] as JSONL so BM25 and dense retrieval share one file.
    Check the field names against the downloaded file before trusting this description.
    """
    raise NotImplementedError("TODO (Diya)")


def load_ibm_argq(path: str):
    """TODO (Diya). The 71 motions with stance labels are also the retrieval-stance test set,
    so keep the motion text and the stance on every passage you create from this file."""
    raise NotImplementedError("TODO (Diya)")
