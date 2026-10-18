from argubot.retrieval.fallback import FallbackRetriever
from argubot.schemas import Passage


class Fixed:
    def __init__(self, passages):
        self.passages = passages

    def retrieve(self, query, k=10):
        return self.passages[:k]


STRONG = [Passage("args.me strong", score=12.0)]
WEAK = [Passage("args.me weak", score=3.0)]
IBM = [Passage("ibm answer", score=5.0)]


def test_strong_primary_result_is_used_and_backup_ignored():
    r = FallbackRetriever(Fixed(STRONG), Fixed(IBM), min_score=8.0)
    assert r.retrieve("q")[0].text == "args.me strong" and r.fallbacks == 0


def test_weak_primary_result_switches_to_the_backup():
    r = FallbackRetriever(Fixed(WEAK), Fixed(IBM), min_score=8.0)
    assert r.retrieve("q")[0].text == "ibm answer"
    assert (r.calls, r.fallbacks) == (1, 1)


def test_score_exactly_at_the_threshold_counts_as_strong():
    assert FallbackRetriever(Fixed([Passage("x", score=8.0)]), Fixed(IBM), 8.0).retrieve("q")[0].text == "x"


def test_empty_backup_does_not_erase_the_primary_answer():
    assert FallbackRetriever(Fixed(WEAK), Fixed([]), 8.0).retrieve("q") == WEAK
    assert FallbackRetriever(Fixed([]), Fixed([]), 8.0).retrieve("q") == []
