"""Choose eval.strict.threshold and eval.strict.min_similarity for the strict contradiction rate.

    python scripts/calibrate_strict_metric.py

Uses only data that no system produced, so the choice cannot be tuned towards a result:
  - the 24 hand-labelled claim/passage pairs from notebooks/05_nli_check.ipynb
    (data/nli_check_pairs.jsonl, 12 real contradictions), which measure what the metric catches;
  - the real CMV replies of the 100 test debates, which people rarely contradict themselves in,
    so whatever the metric flags there is mostly false alarms.

The rule, fixed before looking at any system: among the settings that still catch all 12 real
contradictions, take the one with the lowest per-sentence rate on the real replies; ties go to the
lower similarity cut, then the lower threshold. If no setting catches all 12, the most caught wins
first. Copy the two printed values into eval.strict in configs/config.yaml, commit, and only then
run scripts/evaluate.py --strict on the systems.
"""
import argparse

from argubot.config import load_config
from argubot.eval.metrics import (
    EmbeddingSimilarity,
    contradiction_rate,
    human_transcripts,
    is_assertion,
    is_standalone,
)
from argubot.io import read_jsonl
from argubot.nli.scorer import RobertaMNLIScorer, symmetric_contradiction_batch
from argubot.schemas import Dialogue

THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
SIMILARITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--pairs", default="data/nli_check_pairs.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    strict = cfg["eval"]["strict"]
    scorer = RobertaMNLIScorer(strict["nli_model"], max_length=cfg["nli"]["max_length"])
    similarity = EmbeddingSimilarity(strict["similarity_model"])
    print("NLI model:", strict["nli_model"], "| labels:", scorer.model.config.id2label)

    # The labelled pairs. A pair the sentence filters would drop counts as missed.
    labelled = list(read_jsonl(args.pairs))
    pairs = [(p["claim"], p["text"]) for p in labelled]
    passes = [is_assertion(c) and is_standalone(t, strict["min_words"]) for c, t in pairs]
    nli = symmetric_contradiction_batch(scorer, pairs, combine=min)
    sims = similarity(pairs)
    print(f"\nLabelled pairs: {len(pairs)}, real contradictions: {sum(p['contradiction'] for p in labelled)}")
    print(f"{'#':>3} {'real':>5} {'filters':>8} {'sim':>6} {'NLI min':>8}  claim / text")
    for i, (p, ok, s, n) in enumerate(zip(labelled, passes, sims, nli), 1):
        print(f"{i:>3} {'Y' if p['contradiction'] else 'n':>5} {'pass' if ok else 'DROP':>8} {s:>6.2f} {n:>8.2f}  {p['claim'][:45]} / {p['text'][:45]}")

    # The real replies: record every pair the strict metric would look at, score them once.
    dialogues = [Dialogue.from_dict(d) for d in read_jsonl(cfg["paths"]["test_dialogues"])]
    human = human_transcripts(dialogues)
    seen = []
    filters = {"keep_claim": is_assertion, "keep_sentence": lambda s: is_standalone(s, strict["min_words"])}
    contradiction_rate(human, scorer, pair_scores=lambda ps: seen.extend(ps) or [0.0] * len(ps), **filters)
    unique = list(dict.fromkeys(seen))
    print(f"\nReal replies: {len(human)} debates, {len(unique)} claim/sentence pairs after the filters")
    cache = dict(zip(unique, zip(similarity(unique), symmetric_contradiction_batch(scorer, unique, combine=min))))

    rows = []
    for thr in THRESHOLDS:
        for min_sim in SIMILARITIES:
            caught = sum(p["contradiction"] and ok and s >= min_sim and n >= thr for p, ok, s, n in zip(labelled, passes, sims, nli))
            false = sum(not p["contradiction"] and ok and s >= min_sim and n >= thr for p, ok, s, n in zip(labelled, passes, sims, nli))

            def lookup(ps, min_sim=min_sim):
                return [cache[p][1] if cache[p][0] >= min_sim else 0.0 for p in ps]

            res = contradiction_rate(human, scorer, threshold=thr, pair_scores=lookup, **filters)
            rows.append((thr, min_sim, caught, false, res.rate, res.sentence_rate))

    n_real = sum(p["contradiction"] for p in labelled)
    print(f"\n{'threshold':>9} {'min sim':>8} {'caught /' + str(n_real):>10} {'false alarms':>13} {'human per turn':>15} {'human per sent.':>16}")
    for thr, min_sim, caught, false, rate, srate in rows:
        print(f"{thr:>9.1f} {min_sim:>8.1f} {caught:>10} {false:>13} {rate:>15.3f} {srate:>16.3f}")

    best = min(rows, key=lambda r: (-r[2], r[5], r[1], r[0]))
    print(
        f"\nChosen by the rule: threshold {best[0]}, min_similarity {best[1]} "
        f"(catches {best[2]}/{n_real}, {best[3]} false alarms on the labelled pairs, "
        f"real replies {best[4]:.3f} per turn and {best[5]:.3f} per sentence)."
    )
    print(f"Put in configs/config.yaml under eval.strict:\n    threshold: {best[0]}\n    min_similarity: {best[1]}")


if __name__ == "__main__":
    main()
