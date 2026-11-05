# Results: the 0.8B and 4B runs

Both runs use the same 100 scripted test debates (three user turns each, so three bot replies),
the same five systems and the same evaluation. A turn counts as contradicting when a sentence in
the reply contradicts a claim the bot made earlier in that debate; the first reply has nothing to
contradict, so each system has **200 judged turns**. Every interval below is a paired 95% bootstrap
interval over debates, and a star marks one that excludes zero.

- 0.8B: `outputs/transcripts_fix`, scored into `outputs/results_fix` (the run after the 30 Oct trim fix).
- 4B: `outputs/transcripts_4b`, scored into `outputs/results_4b` (job 352497, scored by job 352499).

## Summary

Neither retrieval, the claim log nor the stance filter changes the contradiction rate
significantly, in either model. The claim log does point the right way in both (24 then 17
contradicting turns of 200 at 0.8B, 17 then 9 at 4B), so the honest description is a consistent
but unconfirmed reduction: 200 judged turns per system is too few to confirm an effect this size.
Reranking (D) is the only significant drop, and it also makes the replies shorter and less
assertive than the proposal's 10% guardrail allows. The 4B model fits CMV replies far better
(perplexity 13.29 against 21.64) and its full system contradicts itself significantly less than
the 0.8B one, but on the metric we trust the claim log's effect is no larger with the bigger model.

## Which metric to believe

The **strict metric** is the headline. The original metric (RoBERTa-MNLI, threshold 0.5) flags 85%
of real CMV replies as self-contradicting, about as often as it flags the systems, so it cannot
tell them apart. The strict metric only compares real assertions with self-contained sentences on
the same subject, needs both NLI directions to agree at 0.7, and uses a DeBERTa model that is not
the one system D reranks with. Its two cut-offs were fixed on 24 labelled pairs and the human
replies before it was run on any system. On real CMV replies it flags 5.7% of turns, which is the
reference line in every table.

## Contradiction rates, strict metric

| System | 0.8B per turn | 4B per turn | 0.8B per sentence | 4B per sentence |
| --- | --- | --- | --- | --- |
| A, history only | 0.070 | 0.070 | 0.021 | 0.016 |
| B, + retrieval | 0.120 | 0.085 | 0.034 | 0.022 |
| C1, + claim log | 0.085 | 0.045 | 0.025 | 0.017 |
| C2, + stance filter | 0.070 | 0.075 | 0.020 | 0.029 |
| D, + reranking | 0.040 | 0.005 | 0.018 | 0.003 |
| Real CMV replies | 0.057 | 0.057 | 0.017 | 0.017 |

## The core test: does the claim log help? (C1 against B, strict)

| | 0.8B | 4B |
| --- | --- | --- |
| Per turn | -0.035 [-0.090, +0.015] | -0.040 [-0.090, +0.010] |
| Per sentence | -0.007 [-0.022, +0.008] | -0.004 [-0.019, +0.010] |
| Flagged turns, B then C1 | 24 then 17 of 200 | 17 then 9 of 200 |

In both models the claim log lowers the contradiction rate, by about a third at 0.8B and by almost
half at 4B, but neither interval excludes zero. Measured per sentence the difference nearly
disappears, so part of the per-turn drop at 4B comes from C1 writing shorter replies with fewer
claims (see the guardrails below). The fair summary is a consistent but not significant reduction.

On the original metric the 0.8B claim log looked harmful: C1 had significantly more contradicting
sentences than B (+0.061 [+0.016, +0.107]). At 4B that difference is gone (-0.002 [-0.044, +0.041]).
One reading is that the small model echoes the claims it is shown in garbled ways that RoBERTa
reads as contradictions, and the bigger model does not. The strict metric shows no such effect at
0.8B, so treat this as a hint, not a finding (the next sections test it properly).

## Every component against the system before it (strict)

Each system adds one component to the one before it, so these steps are the proposal's
one-change-at-a-time comparisons (from `scripts/compare_models.py`):

| Step | 0.8B per turn | 4B per turn | 0.8B per sentence | 4B per sentence |
| --- | --- | --- | --- | --- |
| B - A, retrieval | +0.050 [-0.005, +0.110] | +0.015 [-0.035, +0.065] | +0.008 [-0.008, +0.025] | +0.006 [-0.005, +0.019] |
| C1 - B, claim log | -0.035 [-0.090, +0.015] | -0.040 [-0.090, +0.010] | -0.007 [-0.022, +0.008] | -0.004 [-0.019, +0.010] |
| C2 - C1, stance filter | -0.015 [-0.060, +0.025] | +0.030 [-0.010, +0.075] | -0.004 [-0.016, +0.010] | +0.007 [-0.009, +0.024] |
| D - C2, reranking | -0.030 [-0.075, +0.010] | **-0.070 [-0.120, -0.030] \*** | -0.004 [-0.019, +0.010] | **-0.019 [-0.035, -0.006] \*** |

Only one step passes: reranking at 4B. At 0.8B reranking on its own is not significant, though D
is still significantly below B (-0.080 [-0.135, -0.030], from `evaluate.py --baseline B`) once the
whole chain is counted. The stance filter points in opposite directions in the two models, which is
what no effect looks like, and retrieval, if anything, raises the rate slightly in both (not
significantly), which fits the off-topic BM25 evidence noted in `docs/DATA_NOTES.md`.

## Guardrails: are the systems just saying less?

The proposal keeps a change only if it moves no guardrail by more than about 10%. Claims per turn
and words per reply, with the change from the system before:

| System | 0.8B claims | 0.8B words | 4B claims | 4B words |
| --- | --- | --- | --- | --- |
| A | 4.67 | 77.9 | 4.38 | 72.9 |
| B | 4.46 | 77.6 | 4.29 | 72.2 |
| C1 | 4.56 (+2%) | 77.4 (0%) | 3.88 (-10%) | 65.7 (-9%) |
| C2 | 4.33 (-5%) | 74.7 (-3%) | 3.96 (+2%) | 67.2 (+2%) |
| D | **3.19 (-26%)** | **60.8 (-19%)** | **3.37 (-15%)** | 60.2 (-10%) |
| Real CMV replies | 4.05 | 91.2 | 4.05 | 91.2 |

Distinct-2 stays between 0.50 and 0.54 for every system in both runs, against 0.74 for the human
replies, so the generated text is noticeably more repetitive than people's. No step makes it more
repetitive by more than about 1% (B against A at 0.8B, C2 against C1 at 4B), well inside the guardrail.

**System D** is the only system significantly below B at both sizes, and at 4B the reranking step
on its own is significant too. It also cuts claims per turn by 26% (0.8B) and 15% (4B), past the
10% guardrail. Part of its
consistency is therefore saying less, and the reranker uses an NLI model to pick the reply, which
the metric then judges with another NLI model. The human ratings (below) find that at 4B D's
shorter replies still answer the user about as often as B's.

## What the 4B run adds

**Perplexity.** 21.64 (0.8B) against 13.29 (4B) on the same validation set, so the 4B adapter
models CMV replies far better (`docs/DATA_NOTES.md`).

**Contradictions, system by system, 4B minus 0.8B on the same debates (strict):**

| System | 0.8B | 4B | Per turn | Per sentence |
| --- | --- | --- | --- | --- |
| A | 0.070 | 0.070 | +0.000 [-0.050, +0.045] | -0.006 [-0.019, +0.005] |
| B | 0.120 | 0.085 | -0.035 [-0.100, +0.030] | -0.009 [-0.026, +0.007] |
| C1 | 0.085 | 0.045 | -0.040 [-0.090, +0.010] | -0.007 [-0.021, +0.008] |
| C2 | 0.070 | 0.075 | +0.005 [-0.045, +0.055] | +0.004 [-0.013, +0.022] |
| D | 0.040 | 0.005 | **-0.035 [-0.070, -0.005] \*** | **-0.011 [-0.023, -0.001] \*** |

The 4B rates are lower for B, C1 and D, and the 4B claim-log system sits below the real CMV
replies (4.5% against 5.7% of turns), but paired over the same debates only the full system D is
significantly lower than its 0.8B counterpart.

**Does the claim log matter more with the bigger model?** This was the reason for the 4B run. The
claim log's effect (C1 minus B) in each model, and how it changes:

| | Strict per turn | Strict per sentence | Original per turn | Original per sentence |
| --- | --- | --- | --- | --- |
| 0.8B effect | -0.035 [-0.090, +0.015] | -0.007 [-0.022, +0.008] | +0.040 [-0.005, +0.090] | **+0.061 [+0.016, +0.107] \*** |
| 4B effect | -0.040 [-0.090, +0.010] | -0.004 [-0.019, +0.010] | -0.020 [-0.065, +0.025] | -0.002 [-0.044, +0.041] |
| Change, 4B minus 0.8B | -0.005 [-0.080, +0.075] | +0.002 [-0.019, +0.024] | -0.060 [-0.130, +0.010] | **-0.063 [-0.121, -0.005] \*** |

On the strict metric the effect is the same size in both models, so the 4B run does not show the
claim log mattering more with a stronger generator. The one significant change is on the original
metric, where the 0.8B claim log made sentences look more contradictory and the 4B one did not.
That fits the idea that a small model handles the claim slot badly, but it rests on the metric
that also flags 85% of human replies, so it is weak evidence and should be reported as such.

## Human ratings

Two raters per run each scored the same 20 test debates for all five systems (100 items),
blind to the system, on the three questions in `docs/HUMAN_RATINGS.md`. The sheets are in
`data/human_ratings/`. A debate's score is the share of raters who gave it 1. Intervals are
paired 95% bootstraps over the 20 debates.

**Agreement (Cohen's kappa).**

| Question | 0.8B | 4B |
| --- | --- | --- |
| kept_position | 0.64 (85/100 agree) | 0.63 (86/100) |
| answered_counterargument | 0.72 (92/100) | 0.64 (85/100) |
| used_support_correctly | -0.06 (89/100) | 0.31 (70/100) |

The first two questions show substantial agreement at both sizes. used_support_correctly does
not: at 0.8B almost every item is 0 (94 and 95 of 100), so kappa is uninformative, and at 4B it is
only fair. Treat the support scores as indicative only.

**Mean scores (share of debates scored 1).**

| System | 0.8B kept | 0.8B answered | 0.8B support | 4B kept | 4B answered | 4B support |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0.68 | 0.88 | 0.05 | 0.75 | 0.80 | 0.30 |
| B | 0.68 | 0.85 | 0.05 | 0.72 | 0.65 | 0.23 |
| C1 | 0.72 | 0.80 | 0.05 | 0.68 | 0.60 | 0.33 |
| C2 | 0.75 | 0.88 | 0.07 | 0.72 | 0.78 | 0.30 |
| D | 0.70 | 0.75 | 0.05 | 0.88 | 0.70 | 0.35 |

**Paired differences that matter for the claims above.**

| | 0.8B kept | 0.8B answered | 4B kept | 4B answered |
| --- | --- | --- | --- | --- |
| C1 - B, claim log | +0.05 [-0.23, +0.30] | -0.05 [-0.17, +0.05] | -0.05 [-0.12, +0.00] | -0.05 [-0.20, +0.10] |
| C2 - C1, stance filter | +0.03 [-0.20, +0.25] | +0.07 [+0.00, +0.20] | +0.05 [-0.05, +0.17] | **+0.17 [+0.05, +0.35] \*** |
| D - B, whole chain | +0.03 [-0.23, +0.28] | -0.10 [-0.33, +0.10] | +0.15 [-0.07, +0.38] | +0.05 [-0.20, +0.30] |

**What they show.**

- **The claim log.** Humans see no gain in keeping position at either size. Together with the
  strict metric's non-significant drop, nothing supports the claim log improving consistency.
- **Reranking.** At 4B, D keeps its position in 88% of debates against 72% for B, the same
  direction as the metric's one significant result, but not significant with 20 debates. At 4B,
  D answers the user's point about as often as B, so its shorter replies are not just dodging. At
  0.8B, D has the lowest answered score (0.75), also not significant.
- **The stance filter** is the one significant step: at 4B, C2 answers the counterargument more
  often than C1. With this many comparisons and 20 debates, it is a lead, not a finding.
- **Humans see more inconsistency than the metric.** About 25 to 30% of debates fail
  kept_position for every system, while the strict metric flags at most 12% of turns. Humans also
  count concessions that give up the side, which NLI does not look for, so the strict metric is a
  conservative measure of position keeping.
- **4B against 0.8B**, averaged over the five systems on the same debates: support is much better
  (+0.25 [+0.14, +0.35]), answering the counterargument is worse (-0.12 [-0.23, -0.02]), and
  keeping position is unchanged (+0.04 [-0.10, +0.19]). The two runs had different rater pairs,
  so part of these gaps may come from how strict each pair was.
- **Power.** With 20 debates, only differences of roughly 25 points can show as significant.

## Limitations to state

- **Power.** 200 judged turns per system gives intervals about 5 points wide either side, so an
  effect of 3 to 4 points, the size the claim log shows, cannot be confirmed with this test set.
- **The models differ in more than size.** 0.8B was trained for 3 epochs in fp16 and 4B for 1 epoch
  in bf16 (both without warmup; `docs/UPGRADE_TO_4B.md`).
- **Automatic judges.** The strict metric was calibrated on 24 labelled pairs; how many of its flags
  are real is measured by the precision sheet (`scripts/make_precision_sheet.py`). The human
  ratings are the independent check, and they are only 20 debates per system.
- **Human-rating keys.** The keys in `data/human_ratings/` were rebuilt from the sheet script's seed,
  not copied from the cluster (see `data/human_ratings/README.md`).
- **Retrieval quality.** BM25 evidence is often off topic (`docs/DATA_NOTES.md`), which fits
  retrieval not helping here; a dense retriever is built but was not used in these runs.

## Reproduce these tables

```bash
grep -B 1 -A 8 "differences against" outputs/results_fix/eval.log
```

```bash
grep -B 1 -A 8 "differences against" outputs/results_4b/eval.log
```

```bash
python scripts/compare_models.py
```

```bash
python scripts/compare_models.py --metric original
```

```bash
python scripts/rating_differences.py
```
