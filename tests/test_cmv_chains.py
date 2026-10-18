from argubot.data.cmv import prepare_text, cap_words, chains_for_thread, extract_persona_chains

LONG = "This is a reasonably long reply that makes a clear point about the topic at hand."


def post(pid, parent, author, text=LONG, ts=1, distinguished=None):
    return {"id": pid, "parent": parent, "author": author, "text": text, "ts": ts, "distinguished": distinguished}


def thread():
    return [
        post("t3_a", None, "op", "Uniforms are bad. They cost money.", ts=None),
        post("c1", "t3_a", "alice", ts=1),
        post("c2", "c1", "op", "But what about bullying in schools today?", ts=2),
        post("c3", "c2", "alice", ts=3),
        post("c4", "t3_a", "bob", ts=2),  # bob replies once only: not a persona
    ]


def test_persona_needs_two_replies_and_alternates():
    chains = chains_for_thread(thread(), "op")
    assert len(chains) == 1
    assert [t["speaker"] for t in chains[0]] == ["user", "bot", "user", "bot"]


def test_root_with_none_timestamp_is_fine():
    assert chains_for_thread(thread(), "op")  # ts=None on the opening post must not crash


def test_removed_bot_moderator_and_short_comments_break_the_chain():
    for bad in (
        post("c3", "c2", "alice", "[deleted]", ts=3),
        post("c3", "c2", "alice", ts=3, distinguished="moderator"),
        post("c3", "c2", "alice", "lol no", ts=3),
    ):
        posts = [p for p in thread() if p["id"] != "c3"] + [bad]
        assert chains_for_thread(posts, "op") == []


def test_automated_user_is_never_a_persona():
    posts = thread() + [post("d1", "c2", "DeltaBot"), post("d2", "d1", "op", "ok then"), post("d3", "d2", "DeltaBot")]
    assert len(chains_for_thread(posts, "op")) == 1


def test_quotes_and_entities_are_cleaned():
    posts = thread()
    posts[1]["text"] = "&gt; Uniforms are bad\n" + LONG + " Fish &amp; chips."
    text = chains_for_thread(posts, "op")[0][1]["text"]
    assert "Uniforms are bad" not in text and "Fish & chips." in text


def test_max_turns_is_respected():
    posts = thread() + [post("c5", "c3", "op", "Second thought on this matter altogether."), post("c6", "c5", "alice")]
    assert len(chains_for_thread(posts, "op", max_turns=5)[0]) == 5
    assert len(chains_for_thread(posts, "op", max_turns=6)[0]) == 6


def test_bot_reply_is_cut_at_a_sentence_boundary():
    out = cap_words("First one is here. Second one is here. Third one is here.", 7, whole_sentences=True)
    assert out == "First one is here."
    assert cap_words("One two three four five six seven.", 3, whole_sentences=True) is None
    assert cap_words("one two three four five", 3, whole_sentences=False) == "one two three"


def test_long_first_sentence_drops_the_chain():
    posts = thread()
    posts[1]["text"] = " ".join(["word"] * 40) + "."
    assert chains_for_thread(posts, "op", bot_max_words=20) == []


class _Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_extract_splits_by_thread_and_keeps_dialogue_ids_unique():
    def utt(p):
        return _Obj(id=p["id"], reply_to=p["parent"], speaker=_Obj(id=p["author"]), text=p["text"],
                    timestamp=p["ts"], meta={"distinguished": p["distinguished"]})

    def convo(tid):
        posts = [dict(p, id=f"{tid}_{p['id']}", parent=(f"{tid}_{p['parent']}" if p["parent"] else None)) for p in thread()]
        return _Obj(id=f"{tid}_t3_a", meta={"op-userID": "op"}, iter_utterances=lambda posts=posts: iter(utt(p) for p in posts))

    corpus = _Obj(iter_conversations=lambda: iter(convo(f"T{i}") for i in range(20)))
    cfg = {"seed": 1, "data": {"split_ratios": [0.8, 0.1, 0.1], "min_turns": 3, "max_turns": 6, "bot_reply_max_tokens": 150}}
    rows = extract_persona_chains(corpus, cfg)
    assert len(rows) == 20
    assert len({r["dialogue_id"] for r in rows}) == 20
    assert {r["split"] for r in rows} <= {"train", "val", "test"}
    assert all(set(r) == {"dialogue_id", "split", "turns"} for r in rows)


def test_markdown_links_and_emphasis_are_cleaned():
    raw = "Have you seen the [Purple America](https://x.org/map) map? It is *really* big ^(small) ~~no~~."
    assert prepare_text(raw) == "Have you seen the Purple America map? It is really big (small) no."
