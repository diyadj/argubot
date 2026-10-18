import csv

from argubot.data import clean_comment, is_removed, split_by_thread, split_into_passages
from argubot.eval.human_eval import CRITERIA, kappa_per_criterion, make_rating_sheets, mean_scores_by_system
from argubot.schemas import Transcript, TurnRecord


def test_clean_comment_drops_quotes_and_urls():
    raw = "> quoted line\nMy actual point https://example.com/x here.\n&gt; another quote"
    assert clean_comment(raw) == "My actual point here."


def test_removed_comments_are_detected():
    assert is_removed("[deleted]") and is_removed("  ") and not is_removed("hello")


def test_thread_split_has_the_right_sizes_and_no_leakage():
    ids = [f"thread{i}" for i in range(100)]
    mapping = split_by_thread(ids, (0.8, 0.1, 0.1), seed=1)
    counts = {s: list(mapping.values()).count(s) for s in ("train", "val", "test")}
    assert counts == {"train": 80, "val": 10, "test": 10}
    # Two chains from the same thread must land in the same split because they share one lookup.
    repeated = split_by_thread(ids + ids[:5], (0.8, 0.1, 0.1), seed=1)
    assert repeated == mapping


def test_split_is_deterministic_for_a_seed_and_differs_between_seeds():
    ids = [f"t{i}" for i in range(50)]
    assert split_by_thread(ids, seed=3) == split_by_thread(ids, seed=3)
    assert split_by_thread(ids, seed=3) != split_by_thread(ids, seed=4)


def test_passages_are_packed_by_whole_sentences():
    text = "One two three four five. Six seven eight nine ten. Eleven twelve thirteen fourteen fifteen."
    assert len(split_into_passages(text, max_words=8)) == 3
    assert len(split_into_passages(text, max_words=100)) == 1
    assert split_into_passages("", 10) == []


def make_transcripts(system, ids):
    return [
        Transcript(i, system, [TurnRecord("user words", f"bot words {system}", [], [], [], "p")]) for i in ids
    ]


def test_rating_sheets_are_blinded_and_cover_the_same_dialogues(tmp_path):
    ids = [f"d{i}" for i in range(6)]
    by_system = {"A": make_transcripts("A", ids), "D": make_transcripts("D", ids)}
    paths = make_rating_sheets(by_system, str(tmp_path), n_per_system=4, n_raters=2, seed=0)
    assert len(paths) == 3

    key = list(csv.DictReader(open(tmp_path / "key_PRIVATE.csv", encoding="utf-8")))
    assert len(key) == 8
    assert {r["system"] for r in key} == {"A", "D"}

    sheet = open(tmp_path / "rater_1.csv", encoding="utf-8").read()
    assert "system" not in sheet.splitlines()[0]
    assert "bot words A" in sheet  # the dialogue text is there, the label is not
    rows1 = [r["item_id"] for r in csv.DictReader(open(tmp_path / "rater_1.csv", encoding="utf-8"))]
    rows2 = [r["item_id"] for r in csv.DictReader(open(tmp_path / "rater_2.csv", encoding="utf-8"))]
    assert sorted(rows1) == sorted(rows2) and rows1 != rows2  # same items, different order


def test_kappa_and_system_means_from_filled_sheets(tmp_path):
    by_system = {"A": make_transcripts("A", ["d0", "d1"]), "D": make_transcripts("D", ["d0", "d1"])}
    make_rating_sheets(by_system, str(tmp_path), n_per_system=2, n_raters=2, seed=0)

    def fill(name, value_for):
        path = tmp_path / name
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            for r in rows:
                for c in CRITERIA:
                    r[c] = value_for(r["item_id"])
                w.writerow(r)

    key = {r["item_id"]: r["system"] for r in csv.DictReader(open(tmp_path / "key_PRIVATE.csv", encoding="utf-8"))}
    score = lambda item: "1" if key[item] == "D" else "0"
    fill("rater_1.csv", score)
    fill("rater_2.csv", score)

    kappas = kappa_per_criterion(str(tmp_path / "rater_1.csv"), str(tmp_path / "rater_2.csv"))
    assert all(k == 1.0 for k in kappas.values())
    means = mean_scores_by_system(str(tmp_path / "key_PRIVATE.csv"), [str(tmp_path / "rater_1.csv"), str(tmp_path / "rater_2.csv")])
    assert means["D"]["kept_position"] == 1.0 and means["A"]["kept_position"] == 0.0


def test_ibm_argq_rows_become_passages_with_motion_and_stance(tmp_path):
    from argubot.data.argsme import load_ibm_argq

    header = "argument,topic,set,WA,MACE-P,stance_WA,stance_WA_conf\n"
    (tmp_path / "train.csv").write_text(
        header + '"Uniforms  help.",We should wear uniforms,train,0.8,0.3,1,1.0\n'
        '"They cost too much.",We should wear uniforms,train,0.7,0.2,-1,0.9\n',
        encoding="utf-8",
    )
    (tmp_path / "dev.csv").write_text(header + '"",We should wear uniforms,dev,0.5,0.1,1,1.0\n', encoding="utf-8")
    (tmp_path / "test.csv").write_text(header, encoding="utf-8")
    out = load_ibm_argq(str(tmp_path))
    assert [(p.text, p.stance, p.motion) for p in out] == [
        ("Uniforms help.", "PRO", "We should wear uniforms"),
        ("They cost too much.", "CON", "We should wear uniforms"),
    ]
    assert len({p.source_id for p in out}) == 2
