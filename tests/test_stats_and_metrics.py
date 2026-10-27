import pytest

from argubot.eval.metrics import contradiction_rate, distinct_n
from argubot.eval.stats import cohens_kappa, paired_bootstrap_ci
from argubot.nli import DummyNLIScorer
from argubot.schemas import Transcript, TurnRecord


def transcript(dialogue_id, replies, system="X"):
    turns = [TurnRecord("u", r, [], [], [r], "p") for r in replies]
    return Transcript(dialogue_id=dialogue_id, system=system, turns=turns)


def test_kappa_perfect_and_chance_level():
    assert cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]) == pytest.approx(0.0)


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        cohens_kappa([1, 0], [1])


def test_bootstrap_with_a_constant_difference_has_a_tight_interval():
    res = paired_bootstrap_ci([1.0] * 10, [0.0] * 10, n_boot=500)
    assert res.mean_diff == pytest.approx(1.0)
    assert res.ci_low == pytest.approx(1.0) and res.ci_high == pytest.approx(1.0)


def test_bootstrap_interval_contains_the_mean_for_noisy_data():
    a = [0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0]
    b = [0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5]
    res = paired_bootstrap_ci(a, b, n_boot=2000, seed=1)
    assert res.ci_low <= res.mean_diff <= res.ci_high


def test_distinct_n():
    assert distinct_n(["a b a b"], n=2) == pytest.approx(2 / 3)
    assert distinct_n([], n=2) == 0.0


def test_contradiction_rate_skips_the_first_turn_and_flags_a_reversal():
    consistent = transcript(
        "c", ["School uniforms reduce bullying in schools.", "Uniforms also cut morning arguments at home."]
    )
    reversed_ = transcript(
        "r", ["School uniforms reduce bullying in schools.", "School uniforms do not reduce bullying in schools."]
    )
    res = contradiction_rate([consistent, reversed_], DummyNLIScorer())
    assert res.n_turns == 2
    assert res.n_flagged == 1
    assert res.per_dialogue == {"c": 0.0, "r": 1.0}
    assert res.flagged[0]["dialogue_id"] == "r"


def test_contradiction_rate_is_zero_for_single_turn_debates():
    assert contradiction_rate([transcript("s", ["Only one reply here today."])], DummyNLIScorer()).rate == 0.0


def test_human_transcripts_pair_each_persona_reply_with_the_user_turn_before_it():
    from argubot.eval.metrics import human_transcripts
    from argubot.schemas import Dialogue, Turn

    d = Dialogue("d1", [Turn("user", "u1"), Turn("bot", "b1"), Turn("user", "u2"), Turn("bot", "b2")])
    (t,) = human_transcripts([d])
    assert t.system == "human" and [(r.user_text, r.bot_text) for r in t.turns] == [("u1", "b1"), ("u2", "b2")]
    assert human_transcripts([Dialogue("toy", [Turn("user", "u1")])])[0].turns == []
