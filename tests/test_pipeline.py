"""End to end checks that each system adds exactly its own component and nothing more."""
from pathlib import Path

import pytest

from argubot.claims.tracker import ClaimTracker, extract_claims
from argubot.config import load_config
from argubot.io import read_jsonl
from argubot.nli import DummyNLIScorer
from argubot.retrieval import DummyRetriever
from argubot.schemas import Dialogue, Passage
from argubot.systems import ArguBotPipeline, build_systems

ROOT = Path(__file__).resolve().parents[1]
CFG = load_config(str(ROOT / "configs/config.yaml"))
SYSTEMS = build_systems(5)

FIRST = "School uniforms reduce bullying in schools. They also save families money."
BAD = "School uniforms do not reduce bullying in schools at all."
GOOD = "Uniforms also cut morning arguments about clothes at home."


class ScriptedGenerator:
    """Answers the first turn with FIRST. Later turns get a contradicting candidate first."""

    def generate(self, prompt, n=1, seed=None):
        if "Bot:" not in prompt:
            return [FIRST] * n
        return ([BAD, GOOD] + [BAD] * n)[:n]


def make_pipeline(name):
    passages = [Passage.from_dict(d) for d in read_jsonl(str(ROOT / "data/toy/toy_passages.jsonl"))]
    return ArguBotPipeline(
        SYSTEMS[name],
        ScriptedGenerator(),
        CFG,
        retriever=DummyRetriever(passages),
        nli=DummyNLIScorer(),
        tracker_factory=lambda: ClaimTracker(extractor=lambda t: extract_claims(t, use_spacy=False)),
    )


def run(name):
    dialogue = Dialogue.from_dict(
        {
            "dialogue_id": "t1",
            "turns": [
                {"speaker": "user", "text": "Uniforms should be mandatory."},
                {"speaker": "user", "text": "But uniforms limit how students express themselves."},
            ],
        }
    )
    return make_pipeline(name).run_debate(dialogue)


def test_a_sees_neither_claims_nor_evidence():
    t = run("A")
    for turn in t.turns:
        assert turn.claims_in_prompt == [] and turn.passages_in_prompt == []
        assert "[claims]\nnone\n[evidence]\nnone" in turn.prompt


def test_b_adds_evidence_but_no_claims():
    t = run("B")
    assert t.turns[1].passages_in_prompt
    assert t.turns[1].claims_in_prompt == []


def test_c1_adds_the_claim_log_from_turn_two():
    t = run("C1")
    assert t.turns[0].claims_in_prompt == []
    assert t.turns[1].claims_in_prompt == [
        "School uniforms reduce bullying in schools.",
        "They also save families money.",
    ]


def test_c1_keeps_a_contradicting_passage_and_c2_removes_it():
    bad_passage = "do not reduce bullying"
    assert any(bad_passage in p for p in run("C1").turns[1].passages_in_prompt)
    assert not any(bad_passage in p for p in run("C2").turns[1].passages_in_prompt)


def test_c2_still_returns_the_top_three_after_filtering():
    assert len(run("C2").turns[1].passages_in_prompt) == CFG["retrieval"]["top_k_keep"]


def test_only_d_samples_several_candidates_and_picks_the_consistent_one():
    for name in ["A", "B", "C1", "C2"]:
        assert all(len(turn.candidates) == 1 for turn in run(name).turns)
    d = run("D").turns[1]
    assert len(d.candidates) == 5
    assert d.bot_text == GOOD


def test_c2_without_reranking_returns_the_first_candidate_even_if_it_contradicts():
    assert run("C2").turns[1].bot_text == BAD


def test_missing_components_fail_loudly():
    with pytest.raises(ValueError):
        ArguBotPipeline(SYSTEMS["B"], ScriptedGenerator(), CFG)
    with pytest.raises(ValueError):
        ArguBotPipeline(SYSTEMS["D"], ScriptedGenerator(), CFG, retriever=DummyRetriever([]))
