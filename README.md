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
5. `python scripts/train_lora.py` trains the adapter (or use the training notebook).
6. `python scripts/run_systems.py` runs A, B, C1, C2 and D on the test debates.
7. `python scripts/evaluate.py` computes contradiction rate, guardrails and bootstrap intervals.
8. `python scripts/make_rating_sheets.py` writes blinded sheets for the two human raters.
9. `python scripts/print_hparams.py` prints the hyperparameter table for the report.
10. `python scripts/serve_demo.py` opens the chat demo for the presentation (pick a system, see the claim log and retrieved passages next to each reply).

## What is finished and what is not

| Part | State |
| --- | --- |
| Prompt builder, slot blanking, reply-only labels, collator | Done and tested |
| Five-system pipeline, specs, seeds | Done and tested |
| Claim tracker (regex path and spaCy path) | Working, needs tuning on real CMV replies |
| NLI scorer, reranker, both-direction check | Working (RoBERTa path untested until the model is downloaded) |
| BM25 retriever, stance filter | Working |
| Contradiction rate, distinct-2, bootstrap, kappa, rating sheets | Done and tested |
| Chat demo page (`scripts/serve_demo.py`, `web/index.html`) | Works with `--dummy`, real mode untested until the adapter exists |
| Thread split, comment cleaning, passage splitting | Done and tested |
| CMV persona extraction | Done and tested (9,642 dialogues from 2,734 of 3,051 threads) |
| args.me and IBM-ArgQ loading, passage files, fast BM25 index | Done and tested (1.2M args.me passages, 30k IBM-ArgQ) |
| Dense retrieval (MiniLM + FAISS) | Built over all 1.2M args.me passages (8 minutes on an L40S), stored on the cluster scratch, see `docs/DATA_NOTES.md`; the config still uses BM25 |
| IBM-ArgQ backup index and fallback | Built and tested, switched off (`retrieval.use_backup: false`), see `docs/DATA_NOTES.md` |
| Retrieval-stance test on the 71 motions | Done: `scripts/eval_retrieval_stance.py` (BM25 and dense), results in `docs/DATA_NOTES.md` |
| Stance alignment | Done as a tested utility, not wired into the pipeline |
| Training run and the notebooks | TODO (Andrea), see `docs/NOTEBOOK_ADAPTATION.md` |
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
docs/                   interface contract, owners and dates, notebook notes
notebooks/              thin runners (to be written, see docs/NOTEBOOK_ADAPTATION.md)
outputs/                transcripts, results and rating sheets (not committed)
```

Read `docs/INTERFACES.md` before you change anything that another lane calls.
