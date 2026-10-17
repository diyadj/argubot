from argubot.nli import DummyNLIScorer, rerank, symmetric_contradiction_batch
from argubot.retrieval.stance_filter import StanceConsistencyFilter
from argubot.schemas import Passage

CLAIMS = ["School uniforms reduce bullying in schools."]


def test_symmetric_batch_takes_the_larger_direction():
    class OneWay:
        def score(self, premise, hypothesis):
            raise NotImplementedError

        def score_batch(self, pairs):
            from argubot.schemas import NLIScore

            # Only the reversed direction (second argument first) looks like a contradiction.
            return [NLIScore(0.9 if p.startswith("Reversed") else 0.1, 0.0, 0.0) for p, _ in pairs]

    scores = symmetric_contradiction_batch(OneWay(), [("Original", "Reversed")])
    assert scores == [0.9]


def test_rerank_prefers_the_candidate_that_does_not_contradict():
    candidates = [
        "School uniforms do not reduce bullying in schools at all.",
        "Uniforms also cut morning arguments about clothes at home.",
    ]
    best, scores = rerank(candidates, CLAIMS, DummyNLIScorer())
    assert best == 1
    assert scores[0] > scores[1]


def test_rerank_does_not_reward_empty_replies():
    candidates = ["", "School uniforms do not reduce bullying in schools at all."]
    best, scores = rerank(candidates, CLAIMS, DummyNLIScorer())
    assert scores[0] == float("inf")
    assert best == 1


def test_rerank_without_claims_returns_the_first_candidate():
    assert rerank(["one two three four five six", "other reply words go here now"], [], DummyNLIScorer())[0] == 0


def test_stance_filter_drops_contradicting_passages_only():
    keep = Passage("Uniforms also save families money every year.")
    drop = Passage("School uniforms do not reduce bullying in schools.")
    out = StanceConsistencyFilter(DummyNLIScorer(), 0.5).filter([drop, keep], CLAIMS)
    assert out == [keep]


def test_stance_filter_is_a_no_op_without_claims():
    ps = [Passage("anything")]
    assert StanceConsistencyFilter(DummyNLIScorer()).filter(ps, []) == ps
