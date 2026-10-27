# ArguBot

A retrieval-augmented, consistency-tracking debate dialogue system. CS425, group G2T3.

ArguBot argues one side of a debate over several turns, grounds each reply in retrieved arguments, and keeps a log of its own earlier claims so it does not contradict itself. The hypothesis is that a generator conditioned on that claim log contradicts itself less than one that only sees the conversation history.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                  # all tests should pass
python scripts/run_systems.py --dummy   # fake components, checks the wiring end to end
python scripts/evaluate.py --dummy      # prints a results table (numbers are meaningless)
python scripts/serve_demo.py --dummy    # chat demo at http://127.0.0.1:8000 (fake replies)
```

For real runs also install the ML stack with `pip install -r requirements.txt` and `python -m spacy download en_core_web_sm`. Colab already has torch.

The `--dummy` runs use fake components and the toy data in `data/toy`. They only prove that the pieces fit together. Never put their numbers in the report.

## The five systems

Each system adds exactly one component to the one before it, so any difference between neighbours has one cause.

| System | Configuration | Question it answers |
| --- | --- | --- |
| A | Conversation history only | Where does the fine-tuned model land on its own? |
| B | A plus retrieved supporting arguments | Does retrieval make replies better grounded? |
| C1 | B plus the claim log | Does the claim log reduce contradictions? (the core test, C1 against B) |
| C2 | C1 plus the stance-consistency filter | Does removing clashing passages help further? |
| D | C2 plus 5-candidate reranking | Does choosing the most consistent reply help further? |

## Running the whole project

Run these from the repo root. This is also the draft of the report's "how to run" section, so keep it accurate.

1. `python scripts/download_data.py --all` downloads CMV, args.me and IBM-ArgQ (about 250 MB).
2. `python scripts/build_cmv_chains.py` builds persona dialogues, split by thread.
3. `python scripts/build_indexes.py` builds the passage files and the BM25 index (about 4 minutes).
4. `python scripts/build_training_data.py` fills the claim and evidence slots for training.
5. `python scripts/make_test_debates.py` picks the 100 scripted test debates.
6. `python scripts/train_lora.py` trains the adapter (or use `notebooks/02_train.ipynb`).
7. `python scripts/run_systems.py` runs A, B, C1, C2 and D on the test debates. Add `--limit 3 --out outputs/check3` for a quick check.
8. `python scripts/evaluate.py --human` computes contradiction rate, guardrails and bootstrap intervals against system A, with the real CMV replies as a reference row. Add `--baseline B` for the core test (C1 against B) and `--threshold 0.8` for the sensitivity check. Flagged contradiction pairs go to `outputs/results/flagged_<system>.jsonl`. `--strict` uses the strict metric instead (only real assertions against self-contained sentences on the same subject, both NLI directions agreeing, a different NLI model from the reranker's); its two cut-offs come from `python scripts/calibrate_strict_metric.py`, which uses no system output, and `python scripts/make_precision_sheet.py` builds a blinded sheet to measure how many of its flags are real.
9. `python scripts/make_rating_sheets.py` writes blinded sheets for the two human raters.
10. `python scripts/score_ratings.py` checks the filled-in sheets and computes Cohen's kappa and per-system scores.
11. `python scripts/print_hparams.py` prints the hyperparameter table for the report.
12. `python scripts/serve_demo.py` opens the chat demo for the presentation (pick a system, see the claim log and retrieved passages next to each reply).

Steps 1 to 5 are already done: the test debates and training examples are committed, and the passages, indexes and trained adapter are on the SCIS cluster scratch. On the cluster, link them in (`docs/DATA_NOTES.md`, "Built files on the SCIS cluster" and "Trained LoRA adapter"), then run step 7 as a batch job with `sbatch scripts/violet_run.sbatch` (each system is saved as it finishes, and resubmitting carries on where it stopped). See `docs/RUN_ON_VIOLET.md` for first-time cluster setup.

## What is finished and what is not

| Part | State |
| --- | --- |
| Prompt builder, slot blanking, reply-only labels, collator | Done and tested |
| Five-system pipeline, specs, seeds | Done and tested |
| Claim tracker (regex path and spaCy path) | Done, tuned by eye on 50 real CMV replies (`docs/DATA_NOTES.md`) |
| NLI scorer, reranker, both-direction check | Done; real RoBERTa-MNLI checked on 24 labelled pairs, threshold 0.5 kept (`notebooks/05_nli_check.ipynb`) |
| BM25 retriever, stance filter | Working |
| Contradiction rate, distinct-2, bootstrap, kappa, rating sheets, sheet checks | Done and tested |
| Chat demo (`scripts/serve_demo.py`, `notebooks/04_demo.ipynb`) | Works with `--dummy`; real mode loads the trained adapter on the cluster |
| Thread split, comment cleaning, passage splitting | Done and tested |
| CMV persona extraction | Done and tested (9,642 dialogues from 2,734 of 3,051 threads) |
| args.me and IBM-ArgQ loading, passage files, fast BM25 index | Done and tested (1.2M args.me passages, 30k IBM-ArgQ) |
| Dense retrieval (MiniLM + FAISS) | Built over all 1.2M args.me passages (8 minutes on an L40S), stored on the cluster scratch, see `docs/DATA_NOTES.md`; the config still uses BM25 |
| IBM-ArgQ backup index and fallback | Built and tested, switched off (`retrieval.use_backup: false`), see `docs/DATA_NOTES.md` |
| Retrieval-stance test on the 71 motions | Done: `scripts/eval_retrieval_stance.py` (BM25 and dense), results in `docs/DATA_NOTES.md` |
| Stance alignment | Done as a tested utility, not wired into the pipeline |
| Training run | Done: full data, 3 epochs, validation perplexity 19.15; adapter on the cluster scratch (`docs/DATA_NOTES.md`) |
| Notebooks | Done, see `notebooks/README.md` |
| Five-system run, evaluation, human ratings | Next: 3-debate check, then the 100-debate run |
| Training data builder | Done and tested, needs `build_indexes.py` first |
| Real download links | Done: `download_data.py --all` |

## Layout

```
configs/config.yaml     every setting, nothing hardcoded elsewhere
src/argubot/            the package (prompt, claims, retrieval, nli, generate, systems, eval, data)
scripts/                one command per stage of the project
web/                    the chat demo: index.html is built from the React source in web/app/
                        (edit UI: cd web/app, npm install, npm run dev; then npm run build to update index.html)
tests/                  pytest suite, run it before every push
data/toy/               tiny files for the --dummy runs
docs/                   interface contract, data notes for the report, cluster guides
notebooks/              thin runners for Colab and the cluster (training, run + evaluate, demo, NLI check)
outputs/                transcripts, results and rating sheets (not committed)
```

Read `docs/INTERFACES.md` before you change anything that another lane calls.
