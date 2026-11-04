"""Compare the 0.8B and 4B runs on the same 100 test debates, with paired bootstrap intervals.

    python scripts/compare_models.py
    python scripts/compare_models.py --metric original

Uses only files evaluate.py already wrote (transcripts and flagged pairs), so it runs on the
login node in seconds, with no GPU and no NLI model. It prints three tables:

  1. each component against the system just before it (B-A, C1-B, C2-C1, D-C2), in both runs,
     which is the one-change-at-a-time comparison the proposal describes;
  2. each system's contradiction rate in both runs, and the difference (4B minus 0.8B);
  3. the claim log's effect (C1 minus B) in each run, and whether it is bigger in the 4B run.

Before trusting anything it checks that the rebuilt rates match evaluate.py's own totals.
The numbers are saved to the 4B results folder as model_comparison<tag>.json.
"""
import argparse
import json
import sys
from pathlib import Path

from argubot.eval.compare import Difference, effect_change, paired_difference, per_dialogue_rates
from argubot.io import read_jsonl
from argubot.schemas import Transcript

TAGS = {"strict": "_strict", "original": "", "thr0.8": "_thr0.8"}


def load_run(transcripts_dir: str, results_dir: str, tag: str, systems):
    metrics_path = Path(results_dir) / f"metrics{tag}.json"
    totals = json.loads(metrics_path.read_text()) if metrics_path.is_file() else {}
    runs = {}
    for s in systems:
        transcripts = [Transcript.from_dict(d) for d in read_jsonl(str(Path(transcripts_dir) / f"{s}.jsonl"))]
        flagged_path = Path(results_dir) / f"flagged{tag}_{s}.jsonl"
        if not flagged_path.is_file():
            sys.exit(f"Missing {flagged_path}. Run the evaluation for this metric first.")
        rates = per_dialogue_rates(transcripts, read_jsonl(str(flagged_path)))
        expected = totals.get(s, {}).get("contradiction_rate")
        if expected is not None and abs(expected - rates.pooled_turn_rate) > 1e-9:
            sys.exit(f"{s} in {results_dir}: rebuilt rate {rates.pooled_turn_rate:.4f} does not match "
                     f"evaluate.py's {expected:.4f}. The flagged files and transcripts are from different runs.")
        runs[s] = rates
    return runs


def fmt(d) -> str:
    star = " *" if d.significant else ""
    return f"{d.mean:+.3f} [{d.ci_low:+.3f}, {d.ci_high:+.3f}]{star}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-transcripts", default="outputs/transcripts_fix")
    ap.add_argument("--old-results", default="outputs/results_fix")
    ap.add_argument("--new-transcripts", default="outputs/transcripts_4b")
    ap.add_argument("--new-results", default="outputs/results_4b")
    ap.add_argument("--labels", nargs=2, default=["0.8B", "4B"], metavar=("OLD", "NEW"))
    ap.add_argument("--metric", choices=sorted(TAGS), default="strict")
    ap.add_argument("--systems", nargs="+", default=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--effect", nargs=2, default=["C1", "B"], metavar=("TREATED", "CONTROL"),
                    help="the component whose effect is compared across runs (default: the claim log, C1 against B)")
    args = ap.parse_args()

    tag = TAGS[args.metric]
    old = load_run(args.old_transcripts, args.old_results, tag, args.systems)
    new = load_run(args.new_transcripts, args.new_results, tag, args.systems)
    lo, ln = args.labels
    out = {"metric": args.metric, "labels": args.labels, "neighbours": {}, "systems": {}, "effect": {}}

    steps = list(zip(args.systems[1:], args.systems[:-1]))
    print(f"\n{args.metric} metric: each system against the one before it (* = 95% interval excludes 0)")
    print(f"{'Step':<9}{lo + ' per turn':<27}{ln + ' per turn':<27}{lo + ' per sent.':<27}{ln + ' per sent.'}")
    for later, earlier in steps:
        cells, key = [], f"{later}-{earlier}"
        out["neighbours"][key] = {}
        for label, run in ((lo, old), (ln, new)):
            out["neighbours"][key][label] = {
                unit: vars(paired_difference(getattr(run[later], unit), getattr(run[earlier], unit)))
                for unit in ("turn", "sentence")
            }
        for unit in ("turn", "sentence"):
            for label in (lo, ln):
                cells.append(fmt(Difference(**out["neighbours"][key][label][unit])))
        print(f"{key:<9}{cells[0]:<27}{cells[1]:<27}{cells[2]:<27}{cells[3]}")

    print(f"\n{args.metric} metric: {ln} against {lo}, same debates (* = 95% interval excludes 0)")
    print(f"{'System':<7}{lo + ' turn':>11}{ln + ' turn':>11}  {'Diff per turn':<27}{'Diff per sentence'}")
    for s in args.systems:
        dt = paired_difference(new[s].turn, old[s].turn)
        ds = paired_difference(new[s].sentence, old[s].sentence)
        print(f"{s:<7}{old[s].pooled_turn_rate:>11.3f}{new[s].pooled_turn_rate:>11.3f}  {fmt(dt):<27}{fmt(ds)}")
        out["systems"][s] = {"old_turn_rate": old[s].pooled_turn_rate, "new_turn_rate": new[s].pooled_turn_rate,
                             "turn_diff": vars(dt), "sentence_diff": vars(ds)}

    treated, control = args.effect
    print(f"\nEffect of {treated} against {control} in each run, and the change ({ln} effect minus {lo} effect)")
    print(f"{'':<14}{'Per turn':<27}{'Per sentence'}")
    for unit in ("turn", "sentence"):
        e_old, e_new, change = effect_change(
            getattr(new[treated], unit), getattr(new[control], unit),
            getattr(old[treated], unit), getattr(old[control], unit),
        )
        out["effect"][unit] = {"old": vars(e_old), "new": vars(e_new), "change": vars(change)}
    for row, key in ((f"{lo} effect", "old"), (f"{ln} effect", "new"), ("Change", "change")):
        cells = [fmt(Difference(**out["effect"][u][key])) for u in ("turn", "sentence")]
        print(f"{row:<14}{cells[0]:<27}{cells[1]}")

    path = Path(args.new_results) / f"model_comparison{tag}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
