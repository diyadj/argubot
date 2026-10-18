from argubot.claims.tracker import ClaimTracker, extract_claims
from argubot.data.training_data import build_examples, examples_from_dialogue
from argubot.prompt import build_prompt
from argubot.retrieval.base import build_query
from argubot.schemas import Passage, Turn

CFG = {
    "retrieval": {"top_k_retrieve": 4, "top_k_keep": 2},
    "claims": {"max_claims": 5, "dedup_threshold": 0.8},
}


class RecordingRetriever:
    """Returns four passages p0..p3 and remembers every query it was asked."""

    def __init__(self):
        self.queries = []

    def retrieve(self, query, k=10):
        self.queries.append(query)
        return [Passage(text=f"p{i}") for i in range(4)][:k]


def regex_tracker():
    return ClaimTracker(extractor=lambda t: extract_claims(t, use_spacy=False))


DIALOGUE = {
    "dialogue_id": "d1",
    "split": "train",
    "turns": [
        {"speaker": "user", "text": "Uniforms are bad."},
        {"speaker": "bot", "text": "Uniforms reduce bullying in many schools."},
        {"speaker": "user", "text": "They cost too much."},
        {"speaker": "bot", "text": "Families save money on clothing every year."},
    ],
}


def run(retriever=None):
    retriever = retriever or RecordingRetriever()
    return retriever, list(examples_from_dialogue(DIALOGUE, retriever, CFG, tracker_factory=regex_tracker))


def test_one_example_per_bot_turn_with_the_schema_keys():
    _, ex = run()
    assert len(ex) == 2
    assert all(set(e) == {"history", "claims", "passages", "response"} for e in ex)
    assert [e["response"] for e in ex] == [t["text"] for t in DIALOGUE["turns"] if t["speaker"] == "bot"]


def test_first_turn_has_no_claims_and_second_hears_the_first_reply():
    _, ex = run()
    assert ex[0]["claims"] == []
    assert ex[1]["claims"] == ["Uniforms reduce bullying in many schools."]


def test_history_ends_on_the_user_turn_being_answered():
    _, ex = run()
    assert ex[0]["history"] == [{"speaker": "user", "text": "Uniforms are bad."}]
    assert [t["speaker"] for t in ex[1]["history"]] == ["user", "bot", "user"]
    assert ex[1]["history"][-1]["text"] == "They cost too much."


def test_query_and_cutoff_match_the_pipeline():
    retriever, ex = run()
    assert retriever.queries[0] == build_query("Uniforms are bad.", [])
    assert retriever.queries[1] == build_query("They cost too much.", ["Uniforms reduce bullying in many schools."])
    assert ex[0]["passages"] == ["p0", "p1"]  # top_k_retrieve 4, keep 2


def test_stored_example_builds_the_same_prompt_the_pipeline_would():
    _, ex = run()
    e = ex[1]
    prompt = build_prompt([Turn(**t) for t in e["history"]], e["claims"], e["passages"])
    assert "- Uniforms reduce bullying in many schools." in prompt and prompt.endswith("[response]\n")


def test_only_train_and_val_become_examples():
    dialogues = [dict(DIALOGUE, split=s) for s in ("train", "val", "test")]
    out = build_examples(dialogues, RecordingRetriever(), CFG)
    assert set(out) == {"train", "val"} and len(out["train"]) == 2 and len(out["val"]) == 2


def _make_retriever():
    return RecordingRetriever()


def test_parallel_build_matches_the_single_process_build():
    dialogues = [dict(DIALOGUE, dialogue_id=f"d{i}", split="train" if i % 3 else "val") for i in range(12)]
    same = build_examples(dialogues, RecordingRetriever(), CFG)
    from argubot.data.training_data import build_examples_parallel

    par = build_examples_parallel(dialogues, _make_retriever, CFG, workers=2, chunksize=2)
    assert par == same
