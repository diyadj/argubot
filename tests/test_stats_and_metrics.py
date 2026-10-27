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


def test_sentence_rate_counts_flagged_sentences_over_all_scored_sentences():
    ts = [transcript("d1", ["School uniforms reduce bullying in schools.", "School uniforms do not reduce bullying in schools. Lunch is at noon."])]
    res = contradiction_rate(ts, DummyNLIScorer())
    assert (res.n_sentences, res.n_flagged_sentences) == (2, 1) and res.sentence_rate == 0.5
    assert res.per_dialogue_sentence == {"d1": 0.5}


def test_assertion_and_standalone_filters():
    from argubot.eval.metrics import is_assertion, is_standalone

    assert is_assertion("School uniforms reduce bullying in schools.")
    for not_a_claim in [
        "Do uniforms reduce bullying?",
        "Don't let officers abuse their power.",
        "If I punched you in the face every time I saw you.",
        "All you're saying is science shouldn't be dogmatic.",
        "You seem to think putting makeup on is a chore.",
    ]:
        assert not is_assertion(not_a_claim), not_a_claim
    assert is_standalone("Higher minimum wages have not led to job losses.")
    for fragment in ["Rights do not.", "Why would anyone want that to happen at all?", "It is exactly what you said earlier."]:
        assert not is_standalone(fragment), fragment


def test_strict_metric_skips_questions_and_needs_both_directions_and_a_shared_subject():
    from argubot.eval.metrics import strict_options, word_overlap_similarity
    from argubot.nli.scorer import symmetric_contradiction_batch
    from argubot.schemas import NLIScore

    scorer = DummyNLIScorer()
    opts = lambda min_sim: strict_options(scorer, word_overlap_similarity, min_sim, min_words=4)
    real = [transcript("d1", ["School uniforms reduce bullying in schools.", "School uniforms do not reduce bullying in schools."])]
    assert contradiction_rate(real, scorer, **opts(0.2)).rate == 1.0
    assert contradiction_rate(real, scorer, **opts(0.99)).rate == 0.0  # not similar enough to be compared
    asked = [transcript("d2", ["School uniforms reduce bullying in schools.", "Do school uniforms not reduce bullying in schools?"])]
    assert contradiction_rate(asked, scorer).rate == 1.0  # the original metric counts the question
    assert contradiction_rate(asked, scorer, **opts(0.2)).rate == 0.0

    class OneWay:
        def score_batch(self, pairs):
            return [NLIScore(0.9 if p.startswith("A") else 0.1, 0.05, 0.05) for p, _ in pairs]

    assert symmetric_contradiction_batch(OneWay(), [("A x", "B y")]) == [0.9]
    assert symmetric_contradiction_batch(OneWay(), [("A x", "B y")], combine=min) == [0.1]
