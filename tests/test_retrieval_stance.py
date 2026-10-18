from dataclasses import replace

from argubot.eval.retrieval_stance import argument_query_test, motion_query_test
from argubot.schemas import Passage


def corpus():
    """Two motions, four arguments each: two PRO and two CON."""
    ps = []
    for motion in ("m1", "m2"):
        for i, stance in enumerate(("PRO", "PRO", "CON", "CON")):
            ps.append(Passage(f"{motion} arg {i}", stance, source_id=f"{motion}-{i}", motion=motion))
    return ps


class Scripted:
    """Returns whatever list the test hands it, so each metric can be checked by hand."""

    def __init__(self, answer):
        self.answer = answer

    def retrieve(self, query, k=10):
        return [replace(p, score=1.0) for p in self.answer(query)][:k]


def test_motion_query_hit_rate_and_balance():
    ps = corpus()
    by_motion = lambda m: [p for p in ps if p.motion == m]
    rows = motion_query_test(Scripted(lambda q: by_motion(q)), ps, k=4)  # perfect topic match, 2 PRO + 2 CON
    assert [r["hit_at_k"] for r in rows] == [1.0, 1.0] and [r["pro_share"] for r in rows] == [0.5, 0.5]
    wrong = motion_query_test(Scripted(lambda q: by_motion("m2")), ps, k=4)  # always returns m2
    assert wrong[0]["hit_at_k"] == 0.0 and wrong[0]["pro_share"] is None and wrong[1]["hit_at_k"] == 1.0


def test_a_retriever_that_matches_stance_beats_chance():
    ps = corpus()
    same_side = lambda q: [p for p in ps if p.text != q and p.motion == q.split()[0] and p.stance == next(x.stance for x in ps if x.text == q)]
    r = argument_query_test(Scripted(same_side), ps, k=2, n_boot=200)
    assert r["same_stance_rate"] == 1.0 and r["topical_hit_at_k"] == 1.0
    assert abs(r["chance_same_stance_rate"] - 1 / 3) < 1e-9  # 1 other same-side argument out of 3 others in the motion
    assert r["gap_over_chance"] > 0.6


def test_a_topic_only_retriever_sits_on_chance_and_the_query_itself_is_ignored():
    ps = corpus()
    topic_only = lambda q: [p for p in ps if p.motion == q.split()[0]]  # includes the query itself, in file order
    r = argument_query_test(Scripted(topic_only), ps, k=3, n_boot=200)
    assert r["topical_hit_at_k"] == 1.0
    # for every query the 3 results are the other 3 arguments of the motion: exactly the chance baseline
    assert abs(r["same_stance_rate"] - r["chance_same_stance_rate"]) < 1e-9


def test_max_queries_samples_with_a_seed():
    ps = corpus()
    a = argument_query_test(Scripted(lambda q: ps), ps, k=3, max_queries=4, seed=1, n_boot=50)
    b = argument_query_test(Scripted(lambda q: ps), ps, k=3, max_queries=4, seed=1, n_boot=50)
    assert a["n_queries"] == 4 and a == b
