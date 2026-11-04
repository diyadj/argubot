# Results: the 0.8B and 4B runs

Both runs use the same 100 scripted test debates (three user turns each, so three bot replies),
the same five systems and the same evaluation. A turn counts as contradicting when a sentence in
the reply contradicts a claim the bot made earlier in that debate; the first reply has nothing to
contradict, so each system has **200 judged turns**. Every interval below is a paired 95% bootstrap
interval over debates, and a star marks one that excludes zero.

- 0.8B: `outputs/transcripts_fix`, scored into `outputs/results_fix` (the run after the 30 Oct trim fix).
- 4B: `outputs/transcripts_4b`, scored into `outputs/results_4b` (job 352497, scored by job 352499).

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
0.8B, so treat this as a hint, not a finding.

## Every component, strict metric, per turn

Against B, from `evaluate.py --baseline B`:

| Comparison | 0.8B | 4B |
| --- | --- | --- |
| A against B (taking retrieval away) | -0.050 [-0.110, +0.005] | -0.015 [-0.065, +0.035] |
| C2 against B | -0.050 [-0.100, +0.005] | -0.010 [-0.065, +0.050] |
| D against B | **-0.080 [-0.135, -0.030] \*** | **-0.080 [-0.115, -0.045] \*** |

`python scripts/compare_models.py` adds the step by step comparisons (C2 against C1, D against C2)
for both models from the same files.

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

**System D** is the only component with a significant drop in contradictions, at both sizes. It
also cuts claims per turn by 26% (0.8B) and 15% (4B), past the 10% guardrail. Part of its
consistency is therefore saying less, and the reranker uses an NLI model to pick the reply, which
the metric then judges with another NLI model. The human ratings are the way to tell whether D's
shorter replies still answer the user.

## What the 4B run adds

- **Perplexity.** 21.64 (0.8B) against 13.29 (4B) on the same validation set, so the 4B adapter
  models CMV replies far better (`docs/DATA_NOTES.md`).
- **Fewer contradictions in most systems.** B, C1 and D all drop, and the 4B claim-log system (C1,
  4.5% of turns) sits below the real CMV replies (5.7%). `scripts/compare_models.py` gives paired
  intervals for each system across the two models.
- **The same direction for the claim log.** It helps a little in both, a bit more at 4B, and not
  significantly in either. Whether the effect really grows with model size is the change row of
  `compare_models.py`.

## Limitations to state

- **Power.** 200 judged turns per system gives intervals about 5 points wide either side, so an
  effect of 3 to 4 points, the size the claim log shows, cannot be confirmed with this test set.
- **The models differ in more than size.** 0.8B was trained for 3 epochs in fp16 and 4B for 1 epoch
  in bf16 (both without warmup; `docs/UPGRADE_TO_4B.md`).
- **Automatic judges.** The strict metric was calibrated on 24 labelled pairs; how many of its flags
  are real is measured by the precision sheet (`scripts/make_precision_sheet.py`). Human ratings
  (`docs/HUMAN_RATINGS.md`) are the independent check.
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
