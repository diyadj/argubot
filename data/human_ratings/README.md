# Human ratings

The filled-in, blinded rating sheets for both runs (procedure: `docs/HUMAN_RATINGS.md`, results:
the "Human ratings" section of `docs/RESULTS.md`).

| Folder | Transcripts rated | Raters |
| --- | --- | --- |
| `0.8b/` | `outputs/transcripts_fix` (the 0.8B run after the 30 Oct fix) | two, rated separately |
| `4b/` | `outputs/transcripts_4b` | two, rated separately |

Each folder has `rater_1.csv`, `rater_2.csv` and `key_PRIVATE.csv`: 100 items, the five systems
on the same 20 test debates.

**The keys were rebuilt, not copied from the cluster.** `make_rating_sheets.py` picks the debates
and shuffles the items with seed 42, so the key follows from the seed, the system list (A B C1 C2
D) and `data/processed/test_dialogues.jsonl`. For every row of all four sheets, the user turns in
the sheet match the debate the rebuilt key names. Diff these against `outputs/human_eval_fix/` and
`outputs/human_eval_4b/` on the cluster when convenient.

Rating is finished, so the keys no longer need to be kept from anyone.

Reproduce the numbers:

```bash
python scripts/score_ratings.py --dir data/human_ratings/0.8b --results outputs/results_fix
```

```bash
python scripts/score_ratings.py --config configs/config_4b.yaml --dir data/human_ratings/4b
```

```bash
python scripts/rating_differences.py
```
