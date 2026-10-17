from argubot.claims.tracker import ClaimTracker, extract_claims


def regex_tracker(**kwargs):
    return ClaimTracker(extractor=lambda t: extract_claims(t, use_spacy=False), **kwargs)


def test_questions_hedges_and_fragments_are_not_claims():
    text = "Do uniforms help grades? Maybe they help a little. Yes. Uniforms save families a lot of money."
    assert extract_claims(text, use_spacy=False) == ["Uniforms save families a lot of money."]


def test_quoted_opponent_text_is_not_a_claim():
    assert extract_claims('"Uniforms are bad for everyone." Uniforms reduce bullying in schools.', use_spacy=False) == [
        "Uniforms reduce bullying in schools."
    ]


def test_tracker_keeps_only_the_most_recent_claims():
    tracker = regex_tracker(max_claims=5)
    sentences = [
        "Uniforms reduce bullying in many schools.",
        "Families save money on clothing every year.",
        "Morning routines run faster with a fixed outfit.",
        "Dress codes become easier for teachers to enforce.",
        "Students focus more on lessons than fashion.",
        "Parents argue less about what children should wear.",
        "Schools with uniforms report fewer distractions overall.",
    ]
    for s in sentences:
        tracker.update(s)
    assert tracker.claims == sentences[-5:]


def test_a_repeated_claim_is_stored_once_and_moves_to_the_newest_slot():
    tracker = regex_tracker()
    a = "Uniforms reduce bullying in many schools."
    b = "Families save money on clothing every year."
    tracker.update(a)
    tracker.update(b)
    added = tracker.update(a)
    assert added == []
    assert tracker.claims == [b, a]


def test_reset_clears_the_log():
    tracker = regex_tracker()
    tracker.update("Uniforms reduce bullying in many schools.")
    tracker.reset()
    assert tracker.claims == []
