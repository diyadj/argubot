# Who owns what

| Person | Lane | Owns |
| --- | --- | --- |
| Andrea | Generator and systems | Notebooks, Qwen3.5 smoke test and fallback decision, LoRA training, the A/B/C1/C2/D runner, 5-candidate sampling, reranker plug-in |
| Diya | Retrieval | args.me parsing and passage splitting, BM25 and MiniLM + FAISS index, IBM-ArgQ backup index, stance filter, retrieval-stance test on the 71 motions |
| Tricia | Data and claim tracker | CMV persona chains and cleaning, thread-level split, spaCy claim tracker, training set with slots filled, scripted test debates |
| Cassey | NLI and evaluation engineering | RoBERTa-MNLI wrapper, contradiction rate, guardrail metrics, paired bootstrap, Cohen's kappa, blinding script for ratings |
| Lumen | Coordination, human evaluation and write-up | Tracker and weekly check-ins, rating logistics, query and response table, slides, report assembly and page limit, references, README and download-script documentation, peer evaluations |

## Dates

| Date | Milestone |
| --- | --- |
| 19 Oct | Repo set up, stubs published, Qwen3.5 smoke test result |
| 21 Oct | NLI wrapper ready |
| 24 Oct | Retrieval index, claim tracker and training data v1 |
| 31 Oct | Adapter trained |
| 3 Nov | All five systems run on the test debates |
| 5 Nov | Metrics and human ratings done |
| 6 Nov | Error analysis and one improvement round |
| 7 to 9 Nov | Rehearsal and Q&A prep |
| 10 Nov | Presentation |
| 24 Nov | Report draft complete |
| 25 Nov | Report freeze |
| 27 Nov | Submission, peer evaluations |

## Working rules

- Work on your own branch and open a pull request. Another lane's owner reviews anything that touches the interface.
- Run `pytest` before you push. Add a test for any function you finish.
- Write your own report section while the work is fresh. Keep notes in the file where the code lives, then paste them into the report.
- A short check-in each week: what is done, what is blocked, what is next.
