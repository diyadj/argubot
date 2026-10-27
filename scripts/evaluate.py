"""Score every system's transcripts and compare each one to the baseline (owner: Cassey).

    python scripts/evaluate.py --dummy      # wiring check only, numbers are meaningless
    python scripts/evaluate.py --human      # real run, uses the RoBERTa-MNLI scorer
    python scripts/evaluate.py --human --baseline B   # the core test, C1 against B
    python scripts/evaluate.py --threshold 0.8        # sensitivity check, config untouched
    python scripts/evaluate.py --human --strict       # the strict metric (eval.strict in the config)

Writes a JSON file of results and prints a table that can be pasted into the report. Also writes
every flagged contradiction pair per system (flagged_<system>.jsonl) for the error analysis.
--human adds the real CMV replies of the same debates as a reference row: the metric's noise floor.
Runs with a non-default baseline, threshold or metric get their own file names, so nothing is
overwritten. Both the per-turn rate and the per-sentence rate get paired bootstrap intervals.
"""
import argparse
import json
from pathlib import Path

from argubot.config import ensure_dir, load_config
from argubot.eval.metrics import (
    EmbeddingSimilarity,
    claims_per_turn,
    contradiction_rate,
    distinct_n,
    human_transcripts,
    mean_reply_words,
    strict_options,
    word_overlap_similarity,
)
from argubot.eval.stats import paired_bootstrap_ci
from argubot.io import read_jsonl, write_jsonl
from argubot.nli.scorer import DummyNLIScorer, RobertaMNLIScorer
from argubot.schemas import Dialogue, Transcript


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--systems", nargs="+", default=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--baseline", default="A")
    ap.add_argument("--threshold", type=float, default=None, help="override the contradiction threshold")
    ap.add_argument("--strict", action="store_true", help="use the strict metric from eval.strict")
    ap.add_argument("--human", action="store_true", help="add the real CMV replies as a reference row")
    ap.add_argument("--dialogues", default=None, help="debates for --human (default: the test debates)")
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tdir = Path(args.transcripts or cfg["paths"]["transcripts_dir"])
    strict = cfg["eval"]["strict"] if args.strict else None
    if strict and (strict["threshold"] is None or strict["min_similarity"] is None):
        raise SystemExit("eval.strict is not calibrated yet: run scripts/calibrate_strict_metric.py first.")

    if args.dummy:
        print("DUMMY RUN: fake NLI scorer. Numbers are meaningless.")
        scorer = DummyNLIScorer()
    else:
        model = strict["nli_model"] if strict else cfg["nli"]["model"]
        scorer = RobertaMNLIScorer(model, max_length=cfg["nli"]["max_length"])
    options = {}
    if strict:
        similarity = word_overlap_similarity if args.dummy else EmbeddingSimilarity(strict["similarity_model"])
        options = strict_options(scorer, similarity, strict["min_similarity"], strict["min_words"])

    default_thr = strict["threshold"] if strict else cfg["nli"]["contradiction_threshold"]
    thr = default_thr if args.threshold is None else args.threshold
    tag = ("_strict" if strict else "") + ("" if args.threshold is None else f"_thr{thr:g}")
    out_dir = ensure_dir(cfg["paths"]["results_dir"])

    by_system = {name: [Transcript.from_dict(d) for d in read_jsonl(str(tdir / f"{name}.jsonl"))] for name in args.systems}
    if args.human:
        dialogues_path = args.dialogues or ("data/toy/toy_dialogues.jsonl" if args.dummy else cfg["paths"]["test_dialogues"])
        wanted = {t.dialogue_id for t in by_system[args.baseline]}
        dialogues = [Dialogue.from_dict(d) for d in read_jsonl(dialogues_path)]
        by_system["human"] = human_transcripts([d for d in dialogues if d.dialogue_id in wanted])

    results, per_dialogue, per_sentence = {}, {}, {}
    for name, ts in by_system.items():
        cr = contradiction_rate(ts, scorer, threshold=thr, **options)
        replies = [r.bot_text for t in ts for r in t.turns]
        results[name] = {
            "contradiction_rate": cr.rate,
            "turns_scored": cr.n_turns,
            "turns_flagged": cr.n_flagged,
            "sentence_rate": cr.sentence_rate,
            "sentences_scored": cr.n_sentences,
            "sentences_flagged": cr.n_flagged_sentences,
            "flagged_pairs": len(cr.flagged),
            "distinct_2": distinct_n(replies, 2),
            "claims_per_turn": claims_per_turn(ts),
            "mean_reply_words": mean_reply_words(ts),
        }
        per_dialogue[name] = cr.per_dialogue
        per_sentence[name] = cr.per_dialogue_sentence
        write_jsonl(str(out_dir / f"flagged{tag}_{name}.jsonl"), cr.flagged)

    ids = sorted(per_dialogue[args.baseline])
    for name in args.systems:  # the human row is a reference, not a system, so it gets no CI
        if name == args.baseline:
            continue
        for key, per in (("diff_vs_baseline", per_dialogue), ("sentence_diff_vs_baseline", per_sentence)):
            ci = paired_bootstrap_ci(
                [per[name][i] for i in ids],
                [per[args.baseline][i] for i in ids],
                n_boot=cfg["eval"]["n_boot"],
                alpha=cfg["eval"]["ci_alpha"],
                seed=cfg["seed"],
            )
            results[name][key] = {"mean": ci.mean_diff, "ci_low": ci.ci_low, "ci_high": ci.ci_high}

    base_tag = "" if args.baseline == "A" else f"_vs{args.baseline}"
    out = out_dir / f"metrics{base_tag}{tag}.json"
    out.write_text(json.dumps(results, indent=2))  # flat, one key per system: notebook 03 reads it this way

    def diff(d):
        return ("-", "-") if not d else (f"{d['mean']:+.3f}", f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]")

    print(f"\n{'Strict metric' if strict else 'Original metric'}, threshold {thr:g}, differences against {args.baseline}")
    print(
        f"{'System':<8}{'Per turn':>9}{'Diff':>8}{'95% CI':>19}{'Per sent.':>10}{'Diff':>8}{'95% CI':>19}"
        f"{'Distinct-2':>11}{'Claims/turn':>12}{'Words':>7}"
    )
    for name, r in results.items():
        d_turn, ci_turn = diff(r.get("diff_vs_baseline"))
        d_sent, ci_sent = diff(r.get("sentence_diff_vs_baseline"))
        print(
            f"{name:<8}{r['contradiction_rate']:>9.3f}{d_turn:>8}{ci_turn:>19}{r['sentence_rate']:>10.3f}{d_sent:>8}{ci_sent:>19}"
            f"{r['distinct_2']:>11.3f}{r['claims_per_turn']:>12.2f}{r['mean_reply_words']:>7.1f}"
        )
    print(f"\nSaved to {out}, flagged pairs to {out_dir}/flagged{tag}_<system>.jsonl")


if __name__ == "__main__":
    main()
