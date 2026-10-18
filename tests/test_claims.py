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


def test_concession_keeps_only_the_real_point():
    out = extract_claims("Uniforms can be cheaper for some families, but they limit self expression.", use_spacy=False)
    assert out == ["They limit self expression."]


def test_but_without_a_concession_is_left_alone():
    s = "Uniforms reduce bullying, but they do not remove it entirely."
    assert extract_claims(s, use_spacy=False) == [s]


def test_cant_is_not_a_concession():
    s = "Poor families can't afford extra clothes, but schools rarely help them."
    assert extract_claims(s, use_spacy=False) == [s]


def test_semicolon_splits_into_two_claims():
    out = extract_claims("Uniforms save families money; schools see fewer fights.", use_spacy=False)
    assert out == ["Uniforms save families money.", "Schools see fewer fights."]


def test_run_on_is_cut_before_because_or_dropped():
    reason = " ".join(["reason"] * 40)
    cut = extract_claims(f"Uniforms help every student in every school, because {reason}.", use_spacy=False)
    assert cut == ["Uniforms help every student in every school."]
    assert extract_claims(" ".join(["word"] * 50) + ".", use_spacy=False) == []


def test_short_exclamations_are_not_claims_but_i_think_is():
    out = extract_claims("OK, now we're getting somewhere! I think uniforms reduce bullying.", use_spacy=False)
    assert out == ["I think uniforms reduce bullying."]
