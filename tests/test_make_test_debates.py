import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("make_test_debates", Path(__file__).parent.parent / "scripts" / "make_test_debates.py")
mtd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mtd)

LONG = "This is a perfectly reasonable user turn."


def dialogue(did, split="test", users=(LONG, LONG, LONG)):
    turns = []
    for u in users:
        turns += [{"speaker": "user", "text": u}, {"speaker": "bot", "text": "A bot reply goes here for sure."}]
    return {"dialogue_id": did, "split": split, "turns": turns}


def test_only_three_user_turn_test_debates_without_deltas_or_one_liners():
    assert mtd.is_debatable(dialogue("t_0"))
    assert not mtd.is_debatable(dialogue("t_0", users=(LONG, LONG)))
    assert not mtd.is_debatable(dialogue("t_0", users=(LONG, "∆ thanks for that explanation", LONG)))
    assert not mtd.is_debatable(dialogue("t_0", users=(LONG, "!Delta", LONG)))
    assert not mtd.is_debatable(dialogue("t_0", users=(LONG, "lol ok", LONG)))


def test_one_per_thread_test_split_only_and_seeded():
    ds = [dialogue(f"T{i}_{j}") for i in range(30) for j in range(3)] + [dialogue("X_0", split="train")]
    a = mtd.select_debates(ds, 10, seed=1)
    assert len(a) == 10 and len({mtd.thread_of(d) for d in a}) == 10
    assert all(d["split"] == "test" for d in a)
    assert [d["dialogue_id"] for d in a] == [d["dialogue_id"] for d in mtd.select_debates(list(reversed(ds)), 10, seed=1)]
    assert [d["dialogue_id"] for d in a] != [d["dialogue_id"] for d in mtd.select_debates(ds, 10, seed=2)]


def test_asking_for_more_than_exist_returns_what_there_is():
    assert len(mtd.select_debates([dialogue("A_0"), dialogue("B_0")], 100, seed=0)) == 2
