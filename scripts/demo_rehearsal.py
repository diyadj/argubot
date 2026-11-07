"""Rehearse the live demo before presenting it, and rank which debates to show.

    python scripts/demo_rehearsal.py --config configs/config_4b.yaml
    python scripts/demo_rehearsal.py --config configs/config.yaml
    python scripts/demo_rehearsal.py --dummy        # wiring check, meaningless output

Plays every flow in src/argubot/demo_rehearsal.py through ArguBotPipeline.step, the same call the
demo page makes, several times each (different sampling seeds) for each system. Every run is scored
with the strict contradiction metric and a few demo checks (normal-length replies, on-topic
passages, the last reply coming back to the first point, seconds per reply). It prints:

  1. how many runs of each flow were demo ready, per system, with seconds per reply;
  2. the best flow and system pairs for the main demo;
  3. flows where system A slipped and a system with a claim log did not (contrast candidates);
  4. the full transcript of the best run for the top pairs, to read and rehearse from.

Everything, every run included, is saved to outputs/demo_rehearsal/<config>_<time>.json.
On the cluster run it through scripts/violet_demo_test.sbatch, which does both models in one job.
"""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from argubot.config import load_config
from argubot.demo_rehearsal import (
    FLOWS,
    best_trial,
    contrast_candidates,
    rank,
    score_trial,
    summarize,
)
from argubot.eval.metrics import (
    EmbeddingSimilarity,
    contradiction_rate,
    strict_options,
    word_overlap_similarity,
)
from argubot.nli.scorer import DummyNLIScorer, RobertaMNLIScorer
from argubot.schemas import Transcript, Turn
from argubot.systems import ArguBotPipeline, build_systems
from argubot.systems.components import load_components


def strict_judge(cfg, dummy: bool):
    strict = cfg["eval"]["strict"]
    if dummy:
        scorer, similarity = DummyNLIScorer(), word_overlap_similarity
    else:
        scorer = RobertaMNLIScorer(strict["nli_model"], max_length=cfg["nli"]["max_length"])
        similarity = EmbeddingSimilarity(strict["similarity_model"])
    options = strict_options(scorer, similarity, strict["min_similarity"], strict["min_words"])
    return scorer, strict["threshold"], options


def run_flow(pipeline, flow, trial: int):
    tracker = pipeline.tracker_factory()
    history, records, claims_after, seconds = [], [], [], []
    dialogue_id = f"rehearsal-{flow['id']}-{trial}"   # a different seed per trial, the same per rerun
    for i, text in enumerate(flow["turns"]):
        start = time.time()
        records.append(pipeline.step(history, tracker, Turn("user", text), dialogue_id, i))
        seconds.append(time.time() - start)
        claims_after.append(list(tracker.claims))
    return Transcript(dialogue_id, pipeline.spec.name, records), claims_after, seconds


def print_transcript(flow, system, run, score) -> None:
    print(f"\n--- {flow['title']} · system {system} · run {score.trial + 1} "
          f"({'ready' if score.ready else 'not ready'}, callback {score.callback:.2f}, "
          f"{score.mean_seconds:.1f} s per reply) ---")
    for i, (rec, claims) in enumerate(zip(run["transcript"].turns, run["claims_after"])):
        print(f"\nUSER {i + 1}: {rec.user_text}")
        print(f"BOT  {i + 1}: {rec.bot_text}")
        print(f"  claims shown to the bot: {len(rec.claims_in_prompt)} · claim log after: {len(claims)}")
        for c in claims:
            print(f"    - {c}")
        if rec.passages_in_prompt:
            print(f"  top passage: {rec.passages_in_prompt[0][:110]}...")
    for pair in run["flagged"]:
        print(f"  FLAGGED turn {pair['turn'] + 1} ({pair['score']:.2f}): \"{pair['earlier_claim']}\" "
              f"vs \"{pair['new_sentence']}\"")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--systems", nargs="+", default=["A", "C1", "C2", "D"])
    ap.add_argument("--trials", type=int, default=3, help="runs per flow and system, each with its own seed")
    ap.add_argument("--flows", nargs="+", default=None, help="flow ids (default: all)")
    ap.add_argument("--show", type=int, default=3, help="print the best run of this many top pairs")
    ap.add_argument("--dummy", action="store_true", help="fake components: checks the wiring only")
    args = ap.parse_args()

    cfg = load_config(args.config)
    flows = [f for f in FLOWS if args.flows is None or f["id"] in args.flows]
    label = "dummy" if args.dummy else cfg["generator"]["base_model"].split("/")[-1]
    print(f"Demo rehearsal: {label}, systems {args.systems}, {len(flows)} flows x {args.trials} runs", flush=True)

    generator, retriever, nli = load_components(cfg, dummy=args.dummy)
    specs = build_systems(cfg["decoding"]["n_candidates_rerank"])
    pipelines = {s: ArguBotPipeline(specs[s], generator, cfg, retriever=retriever, nli=nli) for s in args.systems}
    scorer, threshold, options = strict_judge(cfg, args.dummy)

    # One untimed turn first, so model warm-up does not count against the first flow.
    pipelines[args.systems[0]].step([], pipelines[args.systems[0]].tracker_factory(),
                                    Turn("user", flows[0]["turns"][0]), "warm-up", 0)

    runs, scores = [], []
    for flow in flows:
        for system in args.systems:
            for trial in range(args.trials):
                transcript, claims_after, seconds = run_flow(pipelines[system], flow, trial)
                cr = contradiction_rate([transcript], scorer, threshold=threshold, **options)
                score = score_trial(
                    flow, system, trial,
                    replies=[r.bot_text for r in transcript.turns],
                    passages_per_turn=[r.passages_in_prompt for r in transcript.turns],
                    claims_after_each_turn=claims_after,
                    flagged_turns=[p["turn"] for p in cr.flagged],
                    seconds=seconds,
                )
                scores.append(score)
                runs.append({"flow": flow["id"], "system": system, "trial": trial, "transcript": transcript,
                             "claims_after": claims_after, "flagged": cr.flagged})
            done = [s for s in scores if s.flow == flow["id"] and s.system == system]
            print(f"  {flow['id']:<15} {system:<3} ready {sum(s.ready for s in done)}/{len(done)}, "
                  f"{sum(s.mean_seconds for s in done) / len(done):.1f} s per reply", flush=True)

    summaries = summarize(scores)
    by_combo = {(s.flow, s.system): s for s in summaries}

    print(f"\n=== {label}: demo-ready runs per flow (of {args.trials}), seconds per reply ===")
    print(f"{'flow':<16}" + "".join(f"{s:>14}" for s in args.systems))
    for flow in flows:
        cells = []
        for s in args.systems:
            c = by_combo[(flow["id"], s)]
            cells.append(f"{c.ready}/{c.trials} {c.seconds:4.1f}s")
        print(f"{flow['id']:<16}" + "".join(f"{x:>14}" for x in cells))

    ranked = rank(summaries)
    print(f"\n=== {label}: best pairs for the main demo ===")
    print(f"{'rank':<5}{'flow':<16}{'sys':<5}{'ready':>6}{'bait ok':>9}{'callback':>10}{'on topic':>10}"
          f"{'claims t1':>11}{'s/reply':>9}")
    for i, c in enumerate(ranked[:8], 1):
        topic = "-" if c.on_topic is None else f"{c.on_topic:.2f}"
        print(f"{i:<5}{c.flow:<16}{c.system:<5}{c.ready:>4}/{c.trials}{c.bait_clean:>7}/{c.trials}"
              f"{c.callback:>10.2f}{topic:>10}{c.claims_after_first:>11.1f}{c.seconds:>9.1f}")

    contrast = contrast_candidates(summaries, scores) if "A" in args.systems else []
    print(f"\n=== {label}: contrast candidates (A slipped, a system with a claim log did not) ===")
    print(", ".join(contrast) if contrast else "none: no flow had A flagged while a claim-log system stayed clean")
    for run in runs:
        if run["system"] == "A" and run["flow"] in contrast:
            seen = set()
            for pair in run["flagged"]:
                key = (pair["turn"], pair["earlier_claim"], pair["new_sentence"])
                if key in seen:
                    continue
                seen.add(key)
                print(f"  {run['flow']} run {run['trial'] + 1}, turn {pair['turn'] + 1}: "
                      f"\"{pair['earlier_claim']}\" vs \"{pair['new_sentence']}\"")

    flow_by_id = {f["id"]: f for f in flows}
    run_by_key = {(r["flow"], r["system"], r["trial"]): r for r in runs}
    print(f"\n=== {label}: best run of the top {min(args.show, len(ranked))} pairs ===")
    for c in ranked[: args.show]:
        t = best_trial(scores, c.flow, c.system)
        print_transcript(flow_by_id[c.flow], c.system, run_by_key[(c.flow, c.system, t.trial)], t)

    out_dir = Path(cfg["paths"]["output_dir"]) / "demo_rehearsal"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(args.config).stem}{'_dummy' if args.dummy else ''}_{time.strftime('%Y%m%d-%H%M')}.json"
    out.write_text(json.dumps({
        "model": label,
        "summaries": [asdict(s) for s in summaries],
        "ranked": [f"{c.flow}/{c.system}" for c in ranked],
        "contrast_candidates": contrast,
        "runs": [{**{k: v for k, v in r.items() if k != "transcript"}, "transcript": r["transcript"].to_dict(),
                  "score": asdict(next(s for s in scores if (s.flow, s.system, s.trial) == (r["flow"], r["system"], r["trial"])))}
                 for r in runs],
    }, indent=2))
    print(f"\nSaved every run to {out}")


if __name__ == "__main__":
    main()
