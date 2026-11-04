"""Comparing two runs: the rebuilt per-debate rates must equal what contradiction_rate computed."""
import pytest

from argubot.eval.compare import effect_change, paired_difference, per_dialogue_rates
from argubot.eval.metrics import contradiction_rate
from argubot.nli.scorer import DummyNLIScorer
from argubot.schemas import Transcript, TurnRecord


def _t(did, replies, system="C1"):
    return Transcript(did, system, [TurnRecord("u", r, [], [], [], "") for r in replies])


TRANSCRIPTS = [
    _t("d1", ["School uniforms reduce bullying in schools.",
              "School uniforms do not reduce bullying in schools. They also cost money.",
              "Uniforms save families money every year."]),
    _t("d2", ["Homework helps students learn.", "Homework helps students learn a lot.", "It is useful."]),
    _t("d3", ["Cities should ban cars downtown.", "Cities should not ban cars downtown. Buses help.",
              "Cities should not ban cars downtown."]),
]


def test_rebuilt_rates_match_the_metric_exactly():
    cr = contradiction_rate(TRANSCRIPTS, DummyNLIScorer())
    assert cr.n_flagged > 0, "the fixture should contain at least one contradiction"
    rebuilt = per_dialogue_rates(TRANSCRIPTS, cr.flagged)
    assert rebuilt.turn == cr.per_dialogue
    assert rebuilt.sentence == cr.per_dialogue_sentence
    assert rebuilt.pooled_turn_rate == pytest.approx(cr.rate)
    assert rebuilt.n_turns == cr.n_turns


def test_a_run_compared_with_itself_differs_by_nothing():
    cr = contradiction_rate(TRANSCRIPTS, DummyNLIScorer())
    r = per_dialogue_rates(TRANSCRIPTS, cr.flagged)
    d = paired_difference(r.turn, r.turn)
    assert d.mean == 0 and not d.significant


def test_runs_over_different_debates_are_refused():
    with pytest.raises(ValueError, match="same debates"):
        paired_difference({"d1": 0.5}, {"d2": 0.5})


def test_a_component_that_helps_more_in_the_new_run_shows_a_negative_change():
    ids = [f"d{i}" for i in range(40)]
    old_b = {i: 0.5 for i in ids}
    old_c1 = {i: 0.5 for i in ids}            # no effect in the old run
    new_b = {i: 0.5 for i in ids}
    new_c1 = {i: 0.0 for i in ids}            # removes every contradiction in the new run
    e_old, e_new, change = effect_change(new_c1, new_b, old_c1, old_b)
    assert e_old.mean == 0 and e_new.mean == -0.5 and change.mean == -0.5
    assert change.significant
