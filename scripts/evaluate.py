"""Score every system's transcripts and compare each one to the baseline (owner: Cassey).

    python scripts/evaluate.py --dummy      # wiring check only, numbers are meaningless
    python scripts/evaluate.py              # real run, uses the RoBERTa-MNLI scorer

Writes a JSON file of results and prints a table that can be pasted into the report.
"""
import argparse
import json
from pathlib import Path

from argubot.config import ensure_dir, load_config
from argubot.eval.metrics import claims_per_turn, contradiction_rate, distinct_n, mean_reply_words
from argubot.eval.stats import paired_bootstrap_ci
from argubot.io import read_jsonl
from argubot.nli.scorer import DummyNLIScorer, RobertaMNLIScorer
from argubot.schemas import Transcript


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--systems", nargs="+", default=["A", "B", "C1", "C2", "D"])
    ap.add_argument("--transcripts", default=None)
    ap.add_argument("--baseline", default="A")
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tdir = Path(args.transcripts or cfg["paths"]["transcripts_dir"])
    if args.dummy:
        print("DUMMY RUN: fake NLI scorer. Numbers are meaningless.")
        scorer = DummyNLIScorer()
    else:
        scorer = RobertaMNLIScorer(cfg["nli"]["model"], max_length=cfg["nli"]["max_length"])

    thr = cfg["nli"]["contradiction_threshold"]
    results, per_dialogue = {}, {}
    for name in args.systems:
        ts = [Transcript.from_dict(d) for d in read_jsonl(str(tdir / f"{name}.jsonl"))]
        cr = contradiction_rate(ts, scorer, threshold=thr)
        replies = [r.bot_text for t in ts for r in t.turns]
        results[name] = {
            "contradiction_rate": cr.rate,
            "turns_scored": cr.n_turns,
            "distinct_2": distinct_n(replies, 2),
            "claims_per_turn": claims_per_turn(ts),
            "mean_reply_words": mean_reply_words(ts),
        }
        per_dialogue[name] = cr.per_dialogue

    ids = sorted(per_dialogue[args.baseline])
    base = [per_dialogue[args.baseline][i] for i in ids]
    for name in args.systems:
        if name == args.baseline:
            continue
        other = [per_dialogue[name][i] for i in ids]
        ci = paired_bootstrap_ci(
            other, base, n_boot=cfg["eval"]["n_boot"], alpha=cfg["eval"]["ci_alpha"], seed=cfg["seed"]
        )
        results[name]["diff_vs_baseline"] = {"mean": ci.mean_diff, "ci_low": ci.ci_low, "ci_high": ci.ci_high}

    out = ensure_dir(cfg["paths"]["results_dir"]) / "metrics.json"
    out.write_text(json.dumps(results, indent=2))

    print(f"\n{'System':<8}{'Contra rate':>12}{'Diff vs ' + args.baseline:>14}{'95% CI':>20}{'Distinct-2':>12}{'Claims/turn':>13}")
    for name in args.systems:
        r = results[name]
        d = r.get("diff_vs_baseline")
        diff = f"{d['mean']:+.3f}" if d else "-"
        ci = f"[{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]" if d else "-"
        print(f"{name:<8}{r['contradiction_rate']:>12.3f}{diff:>14}{ci:>20}{r['distinct_2']:>12.3f}{r['claims_per_turn']:>13.2f}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
