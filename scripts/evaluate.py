"""Score every system's transcripts and compare each one to the baseline (owner: Cassey).

    python scripts/evaluate.py --dummy      # wiring check only, numbers are meaningless
    python scripts/evaluate.py --human      # real run, uses the RoBERTa-MNLI scorer
    python scripts/evaluate.py --human --baseline B   # the core test, C1 against B
    python scripts/evaluate.py --threshold 0.8        # sensitivity check, config untouched

Writes a JSON file of results and prints a table that can be pasted into the report. Also writes
every flagged contradiction pair per system (flagged_<system>.jsonl) for the error analysis.
--human adds the real CMV replies of the same debates as a reference row: the metric's noise floor.
Runs with a non-default baseline or threshold get their own file names, so nothing is overwritten.
"""
import argparse
import json
from pathlib import Path

from argubot.config import ensure_dir, load_config
from argubot.eval.metrics import claims_per_turn, contradiction_rate, distinct_n, human_transcripts, mean_reply_words
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
    ap.add_argument("--threshold", type=float, default=None, help="override nli.contradiction_threshold")
    ap.add_argument("--human", action="store_true", help="add the real CMV replies as a reference row")
    ap.add_argument("--dialogues", default=None, help="debates for --human (default: the test debates)")
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tdir = Path(args.transcripts or cfg["paths"]["transcripts_dir"])
    if args.dummy:
        print("DUMMY RUN: fake NLI scorer. Numbers are meaningless.")
        scorer = DummyNLIScorer()
    else:
        scorer = RobertaMNLIScorer(cfg["nli"]["model"], max_length=cfg["nli"]["max_length"])

    thr = cfg["nli"]["contradiction_threshold"] if args.threshold is None else args.threshold
    thr_tag = "" if args.threshold is None else f"_thr{thr:g}"
    out_dir = ensure_dir(cfg["paths"]["results_dir"])

    by_system = {name: [Transcript.from_dict(d) for d in read_jsonl(str(tdir / f"{name}.jsonl"))] for name in args.systems}
    if args.human:
        dialogues_path = args.dialogues or ("data/toy/toy_dialogues.jsonl" if args.dummy else cfg["paths"]["test_dialogues"])
        wanted = {t.dialogue_id for t in by_system[args.baseline]}
        dialogues = [Dialogue.from_dict(d) for d in read_jsonl(dialogues_path)]
        by_system["human"] = human_transcripts([d for d in dialogues if d.dialogue_id in wanted])

    results, per_dialogue = {}, {}
    for name, ts in by_system.items():
        cr = contradiction_rate(ts, scorer, threshold=thr)
        replies = [r.bot_text for t in ts for r in t.turns]
        results[name] = {
            "contradiction_rate": cr.rate,
            "turns_scored": cr.n_turns,
            "turns_flagged": cr.n_flagged,
            "flagged_pairs": len(cr.flagged),
            "distinct_2": distinct_n(replies, 2),
            "claims_per_turn": claims_per_turn(ts),
            "mean_reply_words": mean_reply_words(ts),
        }
        per_dialogue[name] = cr.per_dialogue
        write_jsonl(str(out_dir / f"flagged{thr_tag}_{name}.jsonl"), cr.flagged)

    ids = sorted(per_dialogue[args.baseline])
    base = [per_dialogue[args.baseline][i] for i in ids]
    for name in args.systems:  # the human row is a reference, not a system, so it gets no CI
        if name == args.baseline:
            continue
        other = [per_dialogue[name][i] for i in ids]
        ci = paired_bootstrap_ci(
            other, base, n_boot=cfg["eval"]["n_boot"], alpha=cfg["eval"]["ci_alpha"], seed=cfg["seed"]
        )
        results[name]["diff_vs_baseline"] = {"mean": ci.mean_diff, "ci_low": ci.ci_low, "ci_high": ci.ci_high}

    base_tag = "" if args.baseline == "A" else f"_vs{args.baseline}"
    out = out_dir / f"metrics{base_tag}{thr_tag}.json"
    out.write_text(json.dumps(results, indent=2))  # flat, one key per system: notebook 03 reads it this way

    print(f"\nThreshold {thr:g}")
    print(f"{'System':<8}{'Contra rate':>12}{'Diff vs ' + args.baseline:>14}{'95% CI':>20}{'Distinct-2':>12}{'Claims/turn':>13}{'Words':>8}")
    for name, r in results.items():
        d = r.get("diff_vs_baseline")
        diff = f"{d['mean']:+.3f}" if d else "-"
        ci = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "-"
        print(
            f"{name:<8}{r['contradiction_rate']:>12.3f}{diff:>14}{ci:>20}"
            f"{r['distinct_2']:>12.3f}{r['claims_per_turn']:>13.2f}{r['mean_reply_words']:>8.1f}"
        )
    print(f"\nSaved to {out}, flagged pairs to {out_dir}/flagged{thr_tag}_<system>.jsonl")


if __name__ == "__main__":
    main()
