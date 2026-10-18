"""Align an args.me stance label to the wording of a debate motion (owner: Diya).

An args.me premise is PRO or CON towards its own conclusion, and conclusions are worded freely:
"The death penalty should be abolished" and "The death penalty should be allowed" are opposite
claims, so a PRO label on the first means the opposite of a PRO label on the second. Before a stance
label can be compared with the bot's side of a debate, it has to be restated relative to ONE motion.

The rule, using the same NLI model as the rest of the project: compare the passage's own conclusion
with the target motion in both directions and take the larger entailment and contradiction score.
    entailment wins and passes the threshold      -> same polarity, keep the label
    contradiction wins and passes the threshold   -> opposite polarity, flip PRO <-> CON
    neither is clear                              -> the passage is about something else, stance None
A conclusion that is only a topic title ("Death Penalty") has no polarity, so it also gives None.

Our reading of the proposal's wording, not a verified definition. It is NOT wired into the pipeline:
CMV dialogues have no explicit motion to align to, and the stance filter (system C2) compares
passage text with the bot's own claims directly, so it does not read stance labels at all. It is
ready for any code that has a motion in hand, such as the IBM-ArgQ evaluation or the demo.
"""
from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence

from argubot.nli.scorer import NLIScorer
from argubot.schemas import Passage

_FLIP = {"PRO": "CON", "CON": "PRO"}

_nlp = None
_nlp_failed = False


def states_a_position(motion: str) -> bool:
    """False for a bare topic title ("Death Penalty", "Capitalism"), which has no polarity to carry over.

    NLI calls "Death Penalty" entailed by "The death penalty should be allowed" with 0.94 confidence,
    so without this check a title would keep its PRO label as if it meant "pro allowing". A motion
    states a position if it contains a verb or auxiliary ("should be abolished", "brings more harm").
    Needs spaCy; if it is missing the check is skipped.
    """
    global _nlp, _nlp_failed
    if _nlp is None and not _nlp_failed:
        try:
            import spacy

            _nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception:
            _nlp_failed = True
    if _nlp is None:
        return True
    return any(t.pos_ in ("VERB", "AUX") for t in _nlp(motion))


def align_stance(
    stance: Optional[str],
    passage_motion: Optional[str],
    target_motion: str,
    scorer: NLIScorer,
    threshold: float = 0.5,
) -> Optional[str]:
    """The stance of a passage towards target_motion, or None if it cannot be told."""
    if stance not in _FLIP or not passage_motion:
        return None
    if passage_motion.strip().lower() == target_motion.strip().lower():
        return stance
    if not states_a_position(passage_motion):
        return None
    forward, backward = scorer.score_batch([(passage_motion, target_motion), (target_motion, passage_motion)])
    entail = max(forward.entailment, backward.entailment)
    contra = max(forward.contradiction, backward.contradiction)
    if contra >= threshold and contra > entail:
        return _FLIP[stance]
    if entail >= threshold:
        return stance
    return None


def align_passages(
    passages: Sequence[Passage], target_motion: str, scorer: NLIScorer, threshold: float = 0.5
) -> List[Passage]:
    """Copies of the passages with `stance` restated towards target_motion (None where unclear)."""
    cache = {}
    out = []
    for p in passages:
        key = (p.stance, p.motion)
        if key not in cache:
            cache[key] = align_stance(p.stance, p.motion, target_motion, scorer, threshold)
        out.append(replace(p, stance=cache[key]))
    return out
