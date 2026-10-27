# Notebooks

Thin runners that import from `src/argubot`.
The cell-by-cell list of changes is in `docs/NOTEBOOK_ADAPTATION.md`.

| Notebook | Status | What it does |
| --- | --- | --- |
| `01_smoke_test.ipynb` | done | Load the model (Qwen3.5, gpt2-medium fallback), confirm `all-linear`, one LoRA step on `data/toy/toy_train_examples.jsonl`, record trainable params + sec/step, generate one reply. |
| `02_train.ipynb` | done | Runs `train()` from `generate/training.py`: load config + examples, train (eval each epoch), save the adapter, print validation perplexity. Falls back to toy data if the real set isn't built yet. |
| `03_run_and_evaluate.ipynb` | done | Runs `scripts/run_systems.py` then `scripts/evaluate.py` for all five systems and renders `metrics.json` as a table. `DUMMY=True` toggle + adapter/passages prerequisite check. |
| `04_demo.ipynb` | done | Interactive loop (10 Nov): one `ArguBotPipeline` (System D) + one `ClaimTracker` across turns, printing the claim log and retrieved passages next to each reply. Has a scripted mini-debate (rehearsal-safe), a live `input()` loop, and a reset cell; `DUMMY=True` for a model-free dry run. |
| `05_nli_check.ipynb` | new | Step 1 of the run plan (Cassey): load real RoBERTa-MNLI, score 24 hand-labelled claim/passage pairs, sweep the contradiction threshold around 0.5, and run `contradiction_rate` on 20 human test debates as a sanity check. Changes no code or config. |

To run on the SCIS GPU cluster instead of Colab, see `docs/SCIS_VIOLET.md`.

Keep logic out of the notebooks. If a cell grows past a screen, it belongs in the package with a test.
