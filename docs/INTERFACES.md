# Interface contract

Everyone codes against these signatures, and each one already has a stub or a working version in `src/argubot`, so nobody has to wait for anyone. If you need to change a signature, tell the group first, then update the code, the tests and this file in the same commit.

| Piece | Call | Returns | Owner |
| --- | --- | --- | --- |
| Claim tracker | `ClaimTracker.update(bot_reply)` and `.claims` | `update` gives the new claims, `.claims` gives up to 5 strings, oldest first | Tricia |
| Retriever | `retrieve(query, k=10)` | List of `Passage`, best first, with `score` set | Diya |
| Stance filter | `StanceConsistencyFilter(scorer).filter(passages, claims)` | The passages that survive | Diya |
| NLI scorer | `score(premise, hypothesis)` and `score_batch(pairs)` | `NLIScore(contradiction, neutral, entailment)` | Cassey |
| Both-direction check | `symmetric_contradiction_batch(scorer, pairs)` | One float per pair, the larger of the two directions | Cassey |
| Reranker | `rerank(candidates, claims, scorer)` | `(best_index, scores)` | Cassey |
| Generator | `generate(prompt, n=1, seed=None)` | List of n reply strings | Andrea |
| Prompt builder | `build_prompt(history, claims, passages)` | The prompt string, shared by training and inference | Everyone (change only by group agreement) |

## File formats

All JSONL, one object per line.

- **Dialogue** (`test_dialogues.jsonl`, `cmv_chains.jsonl`): `{"dialogue_id", "split", "turns": [{"speaker": "user" or "bot", "text"}]}`. At test time only the user turns are replayed.
- **Passage** (`passages.jsonl`): `{"text", "stance": "PRO" or "CON" or null, "source_id", "motion"}`.
- **Training example** (`train_examples.jsonl`, `val_examples.jsonl`): `{"history": [{"speaker", "text"}], "claims": [...], "passages": [...], "response"}`. Claims and passages must come from the real tracker and retriever, replayed turn by turn, so training matches inference.
- **Transcript** (`outputs/transcripts/<system>.jsonl`): written by `run_systems.py`. Each turn keeps the prompt, the claims and passages shown to the model, and every candidate, so any error can be traced to its cause.

## Rules that keep the comparison fair

- Systems A to D use one sample per turn, system D uses five. Everything else (adapter, decoding settings, test debates, seeds) is shared.
- The tracker listens to the bot in every system, but only C1, C2 and D show the claim log to the generator.
- The metrics never use the tracker's cap of five claims, so they can see contradictions the system could not.
- Anything produced with `--dummy` is a wiring check. Never put it in the report.
