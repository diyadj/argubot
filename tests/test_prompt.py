import random

from argubot.prompt import (
    CLAIMS_TAG,
    EVIDENCE_TAG,
    HISTORY_TAG,
    RESPONSE_TAG,
    apply_slot_dropout,
    build_prompt,
    training_example_to_pair,
)
from argubot.schemas import Turn

CFG = {"max_history_turns": 6, "max_passage_words": 80}


def test_blank_slots_are_written_as_none_not_dropped():
    prompt = build_prompt([Turn("user", "Hello there")])
    assert f"{CLAIMS_TAG}\nnone\n{EVIDENCE_TAG}\nnone\n{HISTORY_TAG}\nUser: Hello there\n{RESPONSE_TAG}\n" == prompt


def test_slots_appear_in_order_and_prompt_ends_with_response_tag():
    prompt = build_prompt([Turn("user", "Hi")], claims=["Uniforms help."], passages=["A passage."])
    assert prompt.index(CLAIMS_TAG) < prompt.index(EVIDENCE_TAG) < prompt.index(HISTORY_TAG) < prompt.index(RESPONSE_TAG)
    assert "- Uniforms help." in prompt and "- A passage." in prompt
    assert prompt.endswith(RESPONSE_TAG + "\n")


def test_history_is_trimmed_to_the_most_recent_turns():
    history = [Turn("user" if i % 2 == 0 else "bot", f"turn{i}") for i in range(10)]
    prompt = build_prompt(history, max_history_turns=3)
    assert "turn9" in prompt and "turn7" in prompt
    assert "turn6" not in prompt


def test_long_passages_are_cut_by_words():
    prompt = build_prompt([Turn("user", "x")], passages=["w " * 200], max_passage_words=10)
    assert prompt.count("w") == 10


def test_slot_dropout_extremes():
    rng = random.Random(0)
    assert apply_slot_dropout(["c"], ["p"], rng, 1.0, 1.0) == ([], [])
    assert apply_slot_dropout(["c"], ["p"], rng, 0.0, 0.0) == (["c"], ["p"])


def test_training_pair_keeps_slots_without_rng_and_flattens_the_response():
    example = {
        "history": [{"speaker": "user", "text": "Why?"}],
        "claims": ["Uniforms help."],
        "passages": ["Evidence."],
        "response": "Because  it\nworks.",
    }
    prompt, response = training_example_to_pair(example, CFG)
    assert "- Uniforms help." in prompt and "- Evidence." in prompt
    assert response == "Because it works."


def test_training_pair_can_blank_slots():
    example = {"history": [{"speaker": "user", "text": "Why?"}], "claims": ["c"], "passages": ["p"], "response": "r"}
    prompt, _ = training_example_to_pair(example, CFG, rng=random.Random(0), p_blank_claims=1.0, p_blank_evidence=1.0)
    assert "- c" not in prompt and "- p" not in prompt
