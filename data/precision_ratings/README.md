# Precision ratings for the strict metric

How many of the strict metric's flags are real contradictions (procedure: `scripts/make_precision_sheet.py`,
results: the "Precision of the strict metric" section of `docs/RESULTS.md`).

| Folder | Flags sampled from | Pairs | Labeller |
| --- | --- | --- | --- |
| `0.8b/` | `outputs/results_fix` (the 0.8B run after the 30 Oct fix) | 87 | Tricia, blind to the system |
| `4b/` | `outputs/results_4b` | 66 | Tricia, blind to the system |

Each folder has `precision_sheet.csv` (the filled-in sheet: one flagged claim and sentence per row,
`real_contradiction` 1 or 0, and a reason in `notes`) and `precision_key_PRIVATE.csv` (which system or
the real CMV replies each pair came from). The sheets were made with seed 42 and the default of up to
30 flagged turns per source, one pair per turn. Labelling is finished, so the keys no longer need to be
kept from anyone.

A pair was marked 1 when the two sentences cannot both be true about the same thing, and 0 when they
are compatible, about different things, or the second only clarifies the first. Whether either
sentence is factually true was not judged.

Reproduce the numbers (needs the results folders from the cluster for `metrics_strict.json`):

```bash
python scripts/make_precision_sheet.py --score --dir data/precision_ratings/0.8b --results outputs/results_fix
```

```bash
python scripts/make_precision_sheet.py --score --dir data/precision_ratings/4b --results outputs/results_4b
```
