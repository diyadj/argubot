# Notebooks

Thin runners that import from `src/argubot`.
The cell-by-cell list of changes is in `docs/NOTEBOOK_ADAPTATION.md`.

| Notebook | Status | What it does |
| --- | --- | --- |
| `01_smoke_test.ipynb` | done | Load the model (Qwen3.5, gpt2-medium fallback), confirm `all-linear`, one LoRA step on `data/toy/toy_train_examples.jsonl`, record trainable params + sec/step, generate one reply. |
| `02_train.ipynb` | done | Runs `train()` from `generate/training.py`: load config + examples, train (eval each epoch), save the adapter, print validation perplexity. Falls back to toy data if the real set isn't built yet. |
| `03_run_and_evaluate.ipynb` | todo | Run `scripts/run_systems.py` and `scripts/evaluate.py` for all five systems; show the results table. |
| `04_demo.ipynb` | done | Interactive loop (10 Nov): one `ArguBotPipeline` (System D) + one `ClaimTracker` across turns, printing the claim log and retrieved passages next to each reply. Has a scripted mini-debate (rehearsal-safe), a live `input()` loop, and a reset cell; `DUMMY=True` for a model-free dry run. |

To run on the SCIS GPU cluster instead of Colab, see `docs/SCIS_VIOLET.md`.

Keep logic out of the notebooks. If a cell grows past a screen, it belongs in the package with a test.
