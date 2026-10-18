"""Pick the scripted test debates every system is run on (owner: Tricia).

    python scripts/make_test_debates.py

Reads the test split of cfg["paths"]["cmv_chains"] and writes eval.n_test_debates dialogues to
cfg["paths"]["test_dialogues"], the file scripts/run_systems.py reads. The test split has 970
dialogues, far more than five systems can be run on with a 0.8B model on Colab, and a contradiction
needs several bot turns to show up, so the selection is:

  - test split only (threads never cross splits, see data/cmv.py)
  - three user turns, so each system gives three replies to check against each other
  - no "delta" awards (a poster thanking someone for changing their view is not a debate turn)
    and no user turn under MIN_USER_WORDS words ("lol", "thanks")
  - at most one dialogue per thread, so no opening post is used twice
  - a seeded shuffle of what is left; the first eval.human_eval_dialogues_per_system are the
    subset the human raters see, so keep the file order.
"""
import argparse
import random
from collections import Counter

from argubot.config import load_config
from argubot.io import read_jsonl, write_jsonl

USER_TURNS = 3
MIN_USER_WORDS = 5


def is_debatable(dialogue: dict) -> bool:
    user_texts = [t["text"] for t in dialogue["turns"] if t["speaker"] == "user"]
    if len(user_texts) != USER_TURNS:
        return False
    if any("∆" in t or "delta" in t.lower() for t in user_texts):
        return False
    return all(len(t.split()) >= MIN_USER_WORDS for t in user_texts)


def thread_of(dialogue: dict) -> str:
    return dialogue["dialogue_id"].rsplit("_", 1)[0]


def select_debates(dialogues, n: int, seed: int):
    """Seeded, one per thread. Sorting first makes the result independent of the input order."""
    pool = sorted((d for d in dialogues if d["split"] == "test" and is_debatable(d)), key=lambda d: d["dialogue_id"])
    random.Random(seed).shuffle(pool)
    seen, chosen = set(), []
    for d in pool:
        if thread_of(d) not in seen:
            seen.add(thread_of(d))
            chosen.append(d)
    return chosen[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    n = cfg["eval"]["n_test_debates"]
    chosen = select_debates(read_jsonl(cfg["paths"]["cmv_chains"]), n, cfg["seed"])
    if len(chosen) < n:
        print(f"warning: only {len(chosen)} debates qualify, asked for {n}")
    count = write_jsonl(cfg["paths"]["test_dialogues"], chosen)
    human = cfg["eval"]["human_eval_dialogues_per_system"]
    topics = len({thread_of(d) for d in chosen})
    print(f"wrote {count} test debates ({topics} threads) to {cfg['paths']['test_dialogues']}")
    print(f"the first {human} are the human-rating subset")
    print("user words per turn:", Counter(len(t["text"].split()) // 50 * 50 for d in chosen for t in d["turns"] if t["speaker"] == "user"))


if __name__ == "__main__":
    main()
