"""Automatic metrics (owner: Cassey).

Contradiction rate is the metric the whole hypothesis rests on (C1 versus B), so it is
core, not an extension. Perplexity, distinct-2 and claims per turn are the guardrails that
stop a system from looking consistent by being generic, repetitive or claim-free.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from argubot.claims.tracker import extract_claims
from argubot.nli.scorer import NLIScorer, symmetric_contradiction_batch
from argubot.schemas import Dialogue, Transcript, TurnRecord
from argubot.text import split_sentences


@dataclass
class ContradictionResult:
    rate: float
    n_turns: int
    n_flagged: int
    per_dialogue: Dict[str, float] = field(default_factory=dict)
    flagged: List[dict] = field(default_factory=list)  # pairs for the raters' spot check


def contradiction_rate(
    transcripts: Sequence[Transcript],
    scorer: NLIScorer,
    threshold: float = 0.5,
    extract: Optional[Callable[[str], List[str]]] = None,
) -> ContradictionResult:
    """Share of bot turns after the first where a reply sentence contradicts an earlier bot claim.

    Both NLI directions are checked and the larger score is used. The earlier claims are NOT
    capped at the tracker's five, so the metric sees more than the system does.

    The default extractor is the same one the claim tracker uses. Pass extract=split_sentences
    to compare against every earlier sentence instead. That is the version to use for the
    independent checker in the proposal's extension, since a tracker mistake then cannot hide a
    contradiction from both the system and the metric.
    """
    extract = extract or extract_claims
    n_turns = 0
    n_flagged = 0
    per_dialogue: Dict[str, float] = {}
    flagged: List[dict] = []

    for t in transcripts:
        earlier: List[str] = []
        d_turns = 0
        d_flagged = 0
        for i, rec in enumerate(t.turns):
            sentences = split_sentences(rec.bot_text)
            if i > 0:
                d_turns += 1
                pairs = [(claim, sent) for claim in earlier for sent in sentences]
                scores = symmetric_contradiction_batch(scorer, pairs)
                hits = [(p, s) for p, s in zip(pairs, scores) if s >= threshold]
                if hits:
                    d_flagged += 1
                    for (claim, sent), s in hits:
                        flagged.append(
                            {
                                "dialogue_id": t.dialogue_id,
                                "system": t.system,
                                "turn": i,
                                "earlier_claim": claim,
                                "new_sentence": sent,
                                "score": s,
                            }
                        )
            earlier.extend(extract(rec.bot_text))
        n_turns += d_turns
        n_flagged += d_flagged
        per_dialogue[t.dialogue_id] = (d_flagged / d_turns) if d_turns else 0.0

    rate = n_flagged / n_turns if n_turns else 0.0
    return ContradictionResult(rate, n_turns, n_flagged, per_dialogue, flagged)


def human_transcripts(dialogues: Sequence[Dialogue]) -> List[Transcript]:
    """The real CMV persona replies of the test debates, as transcripts of a system called "human".

    Scored like any system, they give the metric's noise floor: people rarely contradict themselves
    outright, so whatever rate they get is mostly NLI and claim-extraction error. Dialogues
    without bot turns (the toy ones) give empty transcripts.
    """
    out = []
    for d in dialogues:
        turns, user = [], ""
        for t in d.turns:
            if t.speaker == "user":
                user = t.text
            else:
                turns.append(TurnRecord(user, t.text, [], [], [], ""))
        out.append(Transcript(d.dialogue_id, "human", turns))
    return out


def distinct_n(texts: Sequence[str], n: int = 2) -> float:
    """Unique n-grams divided by all n-grams. Low values mean repetitive replies."""
    grams: List[tuple] = []
    for text in texts:
        tokens = text.split()
        grams.extend(zip(*[tokens[i:] for i in range(n)]))
    return len(set(grams)) / len(grams) if grams else 0.0


def claims_per_turn(
    transcripts: Sequence[Transcript], extract: Optional[Callable[[str], List[str]]] = None
) -> float:
    extract = extract or extract_claims
    counts = [len(extract(rec.bot_text)) for t in transcripts for rec in t.turns]
    return sum(counts) / len(counts) if counts else 0.0


def mean_reply_words(transcripts: Sequence[Transcript]) -> float:
    lengths = [len(rec.bot_text.split()) for t in transcripts for rec in t.turns]
    return sum(lengths) / len(lengths) if lengths else 0.0


def perplexity(model, dataloader, ignore_index: int = -100) -> float:
    """Token-weighted perplexity over reply tokens only.

    The base notebook counted pad tokens and the prompt. Here only labels that are not
    ignore_index count, so batches of different lengths are weighted correctly.
    Evaluate with full slots (system C conditions) and note it in the report.
    """
    import torch

    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            n_tokens = int((batch["labels"][:, 1:] != ignore_index).sum().item())
            total_nll += out.loss.item() * n_tokens
            total_tokens += n_tokens
    return math.exp(total_nll / max(total_tokens, 1))
