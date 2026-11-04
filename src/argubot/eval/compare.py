"""Compare two runs (say the 0.8B and the 4B adapter) on the same test debates.

evaluate.py scores the systems of one run against each other. To compare two runs it would need
the per-debate rates, which it does not save, but they can be rebuilt exactly from what it does
save: the transcripts and the flagged pairs (flagged<tag>_<system>.jsonl), since every flagged pair
names its debate, turn and sentence. That needs no NLI model and no GPU.

The per-debate rates follow contradiction_rate in metrics.py: a debate's turn rate is flagged turns
over its turns after the first, and its sentence rate is flagged sentences over all reply sentences
on those turns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from argubot.eval.stats import paired_bootstrap_ci
from argubot.schemas import Transcript
from argubot.text import split_sentences


@dataclass
class RunRates:
    """Per-debate contradiction rates for one system in one run."""

    turn: Dict[str, float]
    sentence: Dict[str, float]
    n_turns: int
    n_flagged_turns: int

    @property
    def pooled_turn_rate(self) -> float:
        return self.n_flagged_turns / self.n_turns if self.n_turns else 0.0


def per_dialogue_rates(transcripts: Iterable[Transcript], flagged: Iterable[dict]) -> RunRates:
    flagged_turns: Dict[str, set] = {}
    flagged_sentences: Dict[str, set] = {}
    for row in flagged:
        did, turn = row["dialogue_id"], int(row["turn"])
        flagged_turns.setdefault(did, set()).add(turn)
        flagged_sentences.setdefault(did, set()).add((turn, row["new_sentence"]))

    turn_rates: Dict[str, float] = {}
    sentence_rates: Dict[str, float] = {}
    n_turns = n_flagged = 0
    for t in transcripts:
        d_turns = max(len(t.turns) - 1, 0)
        d_sentences = sum(len(split_sentences(rec.bot_text)) for rec in t.turns[1:])
        d_flagged = len(flagged_turns.get(t.dialogue_id, set()))
        d_flagged_sentences = len(flagged_sentences.get(t.dialogue_id, set()))
        turn_rates[t.dialogue_id] = d_flagged / d_turns if d_turns else 0.0
        sentence_rates[t.dialogue_id] = d_flagged_sentences / d_sentences if d_sentences else 0.0
        n_turns += d_turns
        n_flagged += d_flagged
    return RunRates(turn_rates, sentence_rates, n_turns, n_flagged)


@dataclass
class Difference:
    mean: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def significant(self) -> bool:
        return self.ci_low > 0 or self.ci_high < 0


def _shared_ids(*rate_maps: Dict[str, float]) -> List[str]:
    ids = set(rate_maps[0])
    for m in rate_maps[1:]:
        if set(m) != ids:
            raise ValueError("The two runs do not cover the same debates, so they cannot be compared pairwise.")
    return sorted(ids)


def paired_difference(
    new: Dict[str, float], old: Dict[str, float], *, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42
) -> Difference:
    """mean(new - old) over debates, with a paired bootstrap interval."""
    ids = _shared_ids(new, old)
    r = paired_bootstrap_ci([new[i] for i in ids], [old[i] for i in ids], n_boot=n_boot, alpha=alpha, seed=seed)
    return Difference(r.mean_diff, r.ci_low, r.ci_high, r.n)


def effect_change(
    new_treated: Dict[str, float],
    new_control: Dict[str, float],
    old_treated: Dict[str, float],
    old_control: Dict[str, float],
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[Difference, Difference, Difference]:
    """How much a component's effect changes between two runs, per debate.

    For the claim log: treated C1, control B. Returns the effect in the old run (C1 - B), the
    effect in the new run, and the change (new effect - old effect), each with an interval. A
    negative change means the component removes more contradictions in the new run.
    """
    ids = _shared_ids(new_treated, new_control, old_treated, old_control)
    old_effect = [old_treated[i] - old_control[i] for i in ids]
    new_effect = [new_treated[i] - new_control[i] for i in ids]
    zeros = [0.0] * len(ids)

    def ci(a: Sequence[float], b: Sequence[float]) -> Difference:
        r = paired_bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha, seed=seed)
        return Difference(r.mean_diff, r.ci_low, r.ci_high, r.n)

    return ci(old_effect, zeros), ci(new_effect, zeros), ci(new_effect, old_effect)
