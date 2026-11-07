"""The demo rehearsal's scoring rules, checked without a model."""
from argubot.demo_rehearsal import (
    BAIT_TURN,
    FLOWS,
    ComboSummary,
    best_trial,
    callback_score,
    contrast_candidates,
    on_topic_share,
    rank,
    score_trial,
    summarize,
)

FLOW = {"id": "uniforms", "title": "School uniforms", "topic": ["uniform"], "turns": ["a", "b", "c"]}
OK_REPLY = " ".join(["word"] * 40)


def _score(flagged=(), replies=(OK_REPLY,) * 3, passages=(["uniforms help"],) * 3, system="C1", trial=0,
           first_claims=("Uniforms reduce bullying in schools.",), last=None):
    replies = list(replies)
    if last is not None:
        replies[-1] = last
    return score_trial(FLOW, system, trial, replies, passages, [list(first_claims)] * 3, flagged, [1.0, 2.0, 3.0])


def test_every_flow_has_three_turns_a_bait_and_topic_words():
    ids = [f["id"] for f in FLOWS]
    assert len(ids) == len(set(ids)) >= 6
    for f in FLOWS:
        assert len(f["turns"]) == BAIT_TURN + 1 and f["topic"]
        assert f["turns"][BAIT_TURN].lower().startswith("so you")


def test_on_topic_share_counts_passages_and_ignores_systems_without_retrieval():
    assert on_topic_share([["School uniforms work", "Taxes are high"], ["Uniform policy"]], ["uniform"]) == 2 / 3
    assert on_topic_share([[], [], []], ["uniform"]) is None


def test_callback_rewards_coming_back_to_the_first_point():
    claims = ["Uniforms reduce bullying in schools."]
    assert callback_score(claims, "As I said, uniforms reduce bullying.") > callback_score(claims, "Taxes are too high.")
    assert callback_score([], "anything") == 0.0


def test_a_run_is_ready_only_when_clean_normal_length_and_on_topic():
    assert _score().ready
    assert not _score(flagged=[1]).ready
    assert not _score(replies=("too short",) * 3).ready
    assert not _score(passages=(["about taxes"],) * 3).ready
    assert _score(system="A", passages=([],) * 3).ready          # A retrieves nothing, so topic does not apply


def test_the_bait_counts_as_resisted_unless_its_turn_is_flagged():
    assert _score(flagged=[1]).bait_clean
    assert not _score(flagged=[BAIT_TURN]).bait_clean


def test_ranking_puts_ready_pairs_first_then_the_bait_then_the_callback_and_leaves_out_a():
    rows = [
        ComboSummary("f1", "C1", 3, 1, 3, 0.9, 1.0, 2.0, 2.0),
        ComboSummary("f2", "D", 3, 3, 3, 0.1, 1.0, 2.0, 6.0),
        ComboSummary("f3", "C2", 3, 3, 3, 0.5, 1.0, 2.0, 3.0),
        ComboSummary("f4", "A", 3, 3, 3, 1.0, None, 2.0, 1.0),
    ]
    assert [(r.flow, r.system) for r in rank(rows)] == [("f3", "C2"), ("f2", "D"), ("f1", "C1")]


def test_best_trial_prefers_a_ready_run_with_a_strong_callback():
    runs = [_score(trial=0, flagged=[1]), _score(trial=1, last="nothing in common here at all " * 5),
            _score(trial=2, last="Uniforms reduce bullying in schools, as I said at the start, and that matters a great deal "
                                "to the kids who would otherwise be singled out for what they wear.")]
    assert best_trial(runs, "uniforms", "C1").trial == 2


def test_contrast_candidates_need_a_slip_from_a_and_a_clean_memory_system():
    scores = [_score(system="A", flagged=[2]), _score(system="C1"), _score(system="D", flagged=[1])]
    assert contrast_candidates(summarize(scores), scores) == ["uniforms"]
    scores = [_score(system="A"), _score(system="C1")]
    assert contrast_candidates(summarize(scores), scores) == []
