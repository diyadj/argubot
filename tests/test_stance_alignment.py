from argubot.retrieval.stance_alignment import align_passages, align_stance, states_a_position
from argubot.schemas import NLIScore, Passage


class Scripted:
    """NLI scorer whose answer depends only on which pair of motions it is asked about."""

    def __init__(self, table):
        self.table = table
        self.calls = 0

    def score(self, premise, hypothesis):
        self.calls += 1
        return self.table.get((premise, hypothesis), NLIScore(0.1, 0.8, 0.1))

    def score_batch(self, pairs):
        return [self.score(p, h) for p, h in pairs]


TARGET = "The death penalty should be allowed"
SAME = "Capital punishment should be permitted"
OPPOSITE = "The death penalty should be abolished"
OTHER = "Uniforms should be mandatory"

NLI = Scripted(
    {
        (SAME, TARGET): NLIScore(0.02, 0.08, 0.90),
        (OPPOSITE, TARGET): NLIScore(0.95, 0.03, 0.02),
        (TARGET, OPPOSITE): NLIScore(0.90, 0.05, 0.05),
    }
)


def test_same_polarity_keeps_the_label_and_opposite_polarity_flips_it():
    assert align_stance("PRO", SAME, TARGET, NLI) == "PRO"
    assert align_stance("PRO", OPPOSITE, TARGET, NLI) == "CON"
    assert align_stance("CON", OPPOSITE, TARGET, NLI) == "PRO"


def test_unrelated_motion_or_missing_information_gives_none():
    assert align_stance("PRO", OTHER, TARGET, NLI) is None
    assert align_stance(None, SAME, TARGET, NLI) is None
    assert align_stance("PRO", None, TARGET, NLI) is None


def test_identical_motion_needs_no_model_call():
    scorer = Scripted({})
    assert align_stance("CON", TARGET.upper(), TARGET, scorer) == "CON" and scorer.calls == 0


def test_align_passages_copies_and_only_asks_once_per_distinct_motion():
    scorer = Scripted(dict(NLI.table))
    ps = [Passage("a", "PRO", motion=OPPOSITE), Passage("b", "PRO", motion=OPPOSITE), Passage("c", "CON", motion=SAME)]
    out = align_passages(ps, TARGET, scorer)
    assert [p.stance for p in out] == ["CON", "CON", "CON"] and [p.stance for p in ps] == ["PRO", "PRO", "CON"]
    assert scorer.calls == 4  # two (stance, motion) pairs x two directions, not three passages


def test_a_bare_topic_title_has_no_polarity_even_if_nli_calls_it_entailed():
    title_nli = Scripted({("Death Penalty", TARGET): NLIScore(0.02, 0.04, 0.94)})
    assert align_stance("PRO", "Death Penalty", TARGET, title_nli) is None
    assert not states_a_position("Capitalism") and not states_a_position("Death Penalty")
    assert states_a_position("The death penalty should be abolished")
    assert states_a_position("Homeopathy brings more harm than good")
