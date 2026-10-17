"""The one prompt builder.

Training, inference and evaluation all call build_prompt, so the format the model is
trained on is exactly the format it sees at test time. If you change the layout here,
retrain the adapter. The base notebook had separate formats for training and generation,
which silently breaks the claim and evidence slots, so keep it in one place.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

from argubot.schemas import Turn
from argubot.text import flatten_whitespace, truncate_words

CLAIMS_TAG = "[claims]"
EVIDENCE_TAG = "[evidence]"
HISTORY_TAG = "[history]"
RESPONSE_TAG = "[response]"
EMPTY_SLOT = "none"


def _bullets(items: Sequence[str]) -> str:
    items = [flatten_whitespace(i) for i in items if i and i.strip()]
    if not items:
        return EMPTY_SLOT
    return "\n".join(f"- {i}" for i in items)


def build_prompt(
    history: Sequence[Turn],
    claims: Sequence[str] = (),
    passages: Sequence[str] = (),
    *,
    max_history_turns: int = 6,
    max_passage_words: int = 80,
) -> str:
    """Layout: claims, evidence, history, then the response tag. The reply follows the final newline.

    A blank slot is written as the word "none" instead of being dropped, so the
    structure stays the same whichever system is running.
    """
    recent = list(history)[-max_history_turns:]
    lines = [f"{'User' if t.speaker == 'user' else 'Bot'}: {flatten_whitespace(t.text)}" for t in recent]
    trimmed = [truncate_words(flatten_whitespace(p), max_passage_words) for p in passages]
    parts = [
        CLAIMS_TAG,
        _bullets(claims),
        EVIDENCE_TAG,
        _bullets(trimmed),
        HISTORY_TAG,
        "\n".join(lines),
        RESPONSE_TAG,
    ]
    return "\n".join(parts) + "\n"


def apply_slot_dropout(
    claims: Sequence[str],
    passages: Sequence[str],
    rng: random.Random,
    p_claims: float,
    p_evidence: float,
) -> Tuple[List[str], List[str]]:
    """Randomly blank each slot while training, so one adapter can serve systems A to D."""
    kept_claims = [] if rng.random() < p_claims else list(claims)
    kept_passages = [] if rng.random() < p_evidence else list(passages)
    return kept_claims, kept_passages


def training_example_to_pair(
    example: Dict[str, Any],
    prompt_cfg: Dict[str, Any],
    rng: Optional[random.Random] = None,
    p_blank_claims: float = 0.0,
    p_blank_evidence: float = 0.0,
) -> Tuple[str, str]:
    """Turn one stored training record into (prompt, response).

    A record looks like {"history": [{"speaker", "text"}], "claims": [...], "passages": [...], "response": "..."}.
    Pass an rng during training to blank slots at random. Leave it out for validation.
    """
    history = [Turn(**t) for t in example["history"]]
    claims = example.get("claims", [])
    passages = example.get("passages", [])
    if rng is not None:
        claims, passages = apply_slot_dropout(claims, passages, rng, p_blank_claims, p_blank_evidence)
    prompt = build_prompt(
        history,
        claims,
        passages,
        max_history_turns=prompt_cfg["max_history_turns"],
        max_passage_words=prompt_cfg["max_passage_words"],
    )
    return prompt, flatten_whitespace(example["response"])
