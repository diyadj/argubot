# Data and claim tracker: notes for the report

Owner: Tricia (CMV chains, claim tracker, training examples, test debates). Retrieval data (args.me, IBM-ArgQ, indexes) is Diya's lane but was built here, so it is covered too. Every number below comes from a real run of the scripts in this repo.

## Datasets

| Dataset | Used for | Size we use |
| --- | --- | --- |
| CMV Winning Arguments (ConvoKit `winning-args-corpus`) | Dialogues for training and testing | 3,051 threads, 293,297 comments |
| args.me v1.0-cleaned (Zenodo 4139439) | Main retrieval index | 382,545 arguments, 1,198,728 passages |
| IBM-ArgQ-Rank-30k (Hugging Face `ibm-research/argument_quality_ranking_30k`) | Backup index, 71-motion stance test | 30,497 arguments, 71 motions |

IBM's own download page no longer exists (404), so the script fetches the same files from IBM's Hugging Face copy. `python scripts/download_data.py --all` downloads all three.

## CMV persona dialogues (`src/argubot/data/cmv.py`)

A persona is a non-poster who replies to the poster at least twice in one thread. Reddit threads are trees, so a dialogue is the longest path that alternates poster, persona, poster, persona, starting at the opening post. The persona's comments are the bot turns and the poster's are the user turns.

- Dropped: deleted and removed comments, moderators (`distinguished == "moderator"`), DeltaBot and AutoModerator, quoted lines, URLs, Reddit markup, and bot replies under 10 words.
- Length: 4 to 6 turns. The proposal says 3 to 6, but two persona replies need at least four turns, so 3 never occurs.
- Bot replies are cut to 150 words at a sentence boundary (words stand in for tokens). If the first sentence alone is longer, that dialogue is dropped. The poster's opening post is cut to 200 words.
- Result: **9,642 dialogues from 2,734 of the 3,051 threads** (4 turns: 4,178; 5 turns: 2,172; 6 turns: 3,292). Bot replies average 89 words.
- Split by thread, 80/10/10, seed 42, assigned before any dialogue is written, so no thread crosses a split (checked: 0 threads in more than one split).

| Split | Dialogues | Threads | Bot turns |
| --- | --- | --- | --- |
| train | 7,690 | 2,192 | 18,014 |
| val | 982 | 267 | 2,282 |
| test | 970 | 275 | 2,280 |

Corpus quirks worth knowing: the opening post and about 400 other comments have no timestamp, so order comes from `reply_to`, never from sorting by time. Some comments are HTML-encoded twice (`&amp;#8710;` for the delta sign), which the first version decoded only once; 1.4% of turns were affected before the fix.

## Claim tracker (`src/argubot/claims/tracker.py`)

Keeps the 5 most recent distinct declarative sentences from the bot's own replies, de-duplicated at word-set Jaccard 0.8. Tuned by reading its output on 50 real CMV bot replies:

- Questions, quotes, hedged sentences ("maybe", "might") and short exclamations are skipped. "I think X" is kept because it is still the speaker's position.
- A concession ("X can be cheaper, but Y") stores only Y.
- A sentence with a semicolon becomes one claim per clause. A sentence over 40 words is cut before "because", or dropped.
- Average claims per reply fell from 4.5 to 3.8 after tuning.

Limits (say these in the report): the tuning was done by eye on one sample, not against labelled claims. Claims that depend on the previous sentence ("That doesn't sound like a utopia to me") are still stored and carry little meaning on their own. If the NLI contradiction scores look noisy, look here first.

## Retrieval data (`src/argubot/data/argsme.py`, `src/argubot/retrieval/`)

- Premises are packed into passages of at most 120 words on sentence boundaries (so they fit the NLI model); text with no sentence breaks is cut at 120 words, and passages under 8 words are dropped. 345,977 of the 382,545 arguments survive.
- args.me and IBM-ArgQ are kept in separate passage files, so the short IBM arguments cannot crowd args.me out of the main index.
- BM25 is an inverted index with the same Okapi scoring as `rank_bm25` (a test compares them). `rank_bm25` took about 19 s per query at this size, an estimated 100 hours for the training set; the index takes about 50 ms, and the saved index loads in under a second.
- A dense retriever (MiniLM + FAISS) is implemented, tested and built over all of args.me: 1,198,728 passages encoded in 471 s on an NVIDIA L40S (about 2,500 passages per second; on CPU it ran at about 25 per second, roughly 13 hours). The index is a flat inner-product index of 384-dimensional vectors, 1.8 GB. A query takes about 50 ms, except the first, which loads the model. `retrieval.method: dense` switches to it; the config still says `bm25` until the group decides.
- BM25 against dense, read by eye on the top 3 for 3 motions and the last user turn of 3 test debates (`build_indexes.py --dense`, then the comparison run on the cluster). On the short motions both find the right debate. On the conversational CMV turns dense is clearly better: a turn about Deaf culture got deaf-versus-blind arguments from dense, and arguments about Calvinism and Disney from BM25. Dense likes short opening statements ("I will be arguing that ...") that say little. Six queries, so a reading, not a measurement.

## IBM-ArgQ backup index (`src/argubot/retrieval/fallback.py`)

Built and wired in, but **switched off** (`retrieval.use_backup: false`). The config says the backup is used "when args.me results are weak", so `FallbackRetriever` uses it when the best args.me BM25 score is below `backup_min_score` (131.4, the 10th percentile over 500 real training queries, from `scripts/calibrate_backup.py`). Only the args.me score decides, because BM25 scores from two corpora cannot be compared.

Why it is off: the weak queries are mostly short conversational turns ("Can you elaborate a bit more on which device you prefer?", about 52 words against 142 on average), which score low because they carry little content, not because args.me lacks the topic. In the 7 weak queries I read, the IBM backup returned worse evidence than args.me every time or nearly (nuclear disarmament for a question about "nice guys", cosmetic surgery for a phone question), since IBM-ArgQ covers only 71 motions. Turning it on would also change the evidence slot, so the training examples would need rebuilding with it.

## Retrieval-stance test on the 71 motions (`scripts/eval_retrieval_stance.py`)

Question: does retrieval hand back passages on the query's side, or only on its topic? Run on IBM-ArgQ, where every argument has a motion and a PRO/CON stance towards it, so the answers are known. The metric definitions are ours (the proposal's wording was not available); they are in `src/argubot/eval/retrieval_stance.py` and easy to change.

- **Motion as query** (71 motions, top 10): BM25 finds the motion's own arguments 99% of the time (dense: 100%). The hits lean PRO (BM25 66%, dense 73%, where 50% is balanced), because PRO arguments tend to restate the motion.
- **Argument as query** (each argument is a query, itself removed, same 3,000 queries for both):

| Retriever | Right motion in top 10 | Same stance as the query | Chance | Gap over chance (95% CI) |
| --- | --- | --- | --- | --- |
| BM25 | 0.749 | 0.748 | 0.500 | +0.248 (+0.239 to +0.257) |
| Dense (MiniLM) | 0.921 | 0.772 | 0.500 | +0.273 (+0.264 to +0.281) |

BM25 on all 30,497 queries gives the same picture (+0.247). So both retrievers pick up the side well above chance, and dense is clearly better at finding the right topic (0.92 against 0.75) but only modestly better at the side. I had expected retrieval to be close to stance-blind; it is not, on this data.

Limits: IBM arguments are short, crowd-written and all about the same 71 motions, and same-side arguments share phrasing ("should be banned"). This says nothing certain about args.me or CMV. I did not check for near-duplicate arguments between query and results. The CIs treat queries as independent. Dense here is built in memory over the 30k short passages, not the full args.me index.

## Stance alignment (`src/argubot/retrieval/stance_alignment.py`)

An args.me label is PRO or CON towards its own conclusion, and conclusions are worded freely, so "PRO" on "The death penalty should be abolished" is the opposite of "PRO" on "...should be allowed". `align_stance` restates a label towards one target motion using RoBERTa-MNLI in both directions: entailment keeps the label, contradiction flips it, otherwise None. Checked on real args.me conclusions against "The death penalty should be allowed": "should be abolished" and "Should Be Illegal" flip to CON, "should exist" and "should remain active" keep PRO. A bare topic title ("Death Penalty") has no polarity, and NLI wrongly called it entailed (0.94), so titles now return None.

Not wired into the pipeline: CMV dialogues have no explicit motion, and the stance filter compares passage text with the bot's claims and never reads stance labels. This is our reading of the proposal's wording, not a verified definition.

## Training examples (`scripts/build_training_data.py`)

One example per bot turn in a train or val dialogue: 18,014 train and 2,282 val. The slots are produced by replaying each dialogue the way `ArguBotPipeline.step` does at inference: the tracker has heard the persona's earlier replies, the query is the latest user turn plus the claim log, and the top 3 of the top 10 BM25 passages are kept. The stance filter is not replayed (it needs the NLI model).

- 46% of examples have an empty claim log (mostly each dialogue's first bot turn) and 32% have the full 5.
- Prompt length with real passages (1.35 tokens per word, an estimate): median about 605 words, **33% of examples are over 1,024 tokens and 2.4% over 1,536**. The training limit is `max_seq_len` 1024, so about a third of prompts lose their oldest tokens.

## Known weakness: the evidence is often off-topic

Reading 12 random training examples, by my own judgement about 4 had clearly relevant evidence, 4 were loosely related and 4 were off-topic (for example, a question about a TV show returned an argument about Iranian social media). CMV topics are wider than args.me's debate topics, and BM25 on a long conversational query matches common words. This is a reading of a small sample, not a measurement. It matters for the A versus B comparison: if the evidence is weak, system B may not beat A, and that would be the retrieval, not the idea. Options for the group: use the dense retriever, or leave the evidence slot empty when the top BM25 score is low.

## Scripted test debates (`scripts/make_test_debates.py`)

The test split has 970 dialogues, too many to run five systems on. The script picks **100 debates**: test split only, exactly three user turns (so each system gives three replies to compare), no delta-award turns and no user turn under 5 words, one per thread, seeded shuffle (seed 42). The human-rating subset is 20 of these, drawn at random with seed 42 by `make_rating_sheets.py`.

## Data files in the repo

`data/processed/` is git-ignored except for three small files: `test_dialogues.jsonl` (the 100 test debates), `train_examples.jsonl.gz` (15 MB) and `val_examples.jsonl.gz` (1.9 MB). After `git pull`, training can start straight away: `read_jsonl` opens `x.jsonl.gz` when only the gzipped copy exists, so the paths in the config work unchanged. The passages (840 MB) and the BM25 index (600 MB) are not committed; `build_indexes.py` rebuilds them in about 4 minutes. If the training examples are rebuilt, gzip the new ones and commit again, since every version stays in the repo history.

## Built files on the SCIS cluster

The passages, the BM25 indexes and the dense index (3.1 GB together) are in `/common/scratch/projectgrps/CS425/CS425G3/argubot_data`, with a `SHA256SUMS` file. `~/argubot` on the cluster links to them, so nothing needs rebuilding. In a fresh clone on the cluster, link them from the repo root:

```
S=/common/scratch/projectgrps/CS425/CS425G3/argubot_data
ln -s $S/processed/passages.jsonl $S/processed/passages_ibm_argq.jsonl data/processed/
for f in bm25_docs.npy bm25_weights.npy bm25_offsets.npy bm25_vocab.json dense ibm_argq; do ln -s $S/indexes/$f data/indexes/$f; done
```

The dense index stores only vectors, in the order of `passages.jsonl`, and `DenseRetriever.load` only checks the count. Never rebuild one of them without the other.

## Trained LoRA adapter on the cluster

The trained LoRA adapter (~62 MB: 43 MB adapter weights + the matching tokenizer) is shared in scratch at `$S/outputs/adapter`, group-readable, with its own `SHA256SUMS`. It is the full-data adapter: `Qwen/Qwen3.5-0.8B-Base` + LoRA (r16, alpha32, all-linear), trained on all 18,014 examples for 3 epochs, validation perplexity 21.64. That number, and the ones below, come from the saved output of the run that made this adapter (`notebooks/02_train.output.21102026132208.ipynb` on the cluster; the adapter files are stamped 21 Oct 16:26, when that run finished). An earlier version of these notes said 19.15, which matches none of that run's numbers. `outputs/` is git-ignored, so link it in on the cluster rather than committing it:

```
S=/common/scratch/projectgrps/CS425/CS425G3/argubot_data
mkdir -p outputs
ln -sfn $S/outputs/adapter outputs/adapter
```

After linking, `scripts/run_systems.py`, `scripts/evaluate.py` and `notebooks/04_demo.ipynb` pick it up with no config change (they read `paths.adapter_dir: outputs/adapter`). It only means anything on top of the same base model, `Qwen/Qwen3.5-0.8B-Base`. To refresh it, retrain with `02_train.ipynb`, then re-copy to `$S/outputs/adapter` and regenerate its `SHA256SUMS`:

```
S=/common/scratch/projectgrps/CS425/CS425G3/argubot_data
cp -r ~/argubot/outputs/adapter "$S/outputs/adapter"
chmod -R g+rX "$S/outputs/adapter"
sha256sum "$S/outputs/adapter"/* > "$S/outputs/adapter/SHA256SUMS"
```

One thing to say in the report about this adapter: it was trained with no learning-rate warmup.
The config said `warmup_ratio: 0.03`, but transformers 5.x renamed that setting to `warmup_steps`,
and the training code at the time skipped settings it did not recognise with a one-line note. The
config now says 0.0 to match what happened, and the code passes warmup through properly.

The run took 10,992 s (about 3.1 h) on an RTX 3090, and the per-epoch losses show the model
overfitting after the first epoch: training loss keeps falling, validation loss is lowest after
epoch 1 and rises after that. The adapter everyone uses is the epoch 3 one.

| Epoch | Training loss | Validation loss | Validation perplexity |
| --- | --- | --- | --- |
| 1 | 2.976 | 2.968 | 19.45 |
| 2 | 2.703 | 3.000 | 20.09 |
| 3 | 2.533 | 3.074 | 21.63 (21.64 measured on the reply tokens afterwards) |

This does not affect the comparison between the five systems, which all share this one adapter,
but it is worth saying in the report, and it is why fewer epochs make sense for the 4B run.

## The 4B adapter

Trained with `configs/config_4b.yaml` (`docs/UPGRADE_TO_4B.md`). It lives in `~/argubot/outputs/adapter_4b`
on the group login, with a copy in `$S/outputs/adapter_4b` in scratch, and carries its own
`training_summary.json` (precision, batch, training time over all jobs, eval loss per epoch,
validation perplexity). It only works on top of `Qwen/Qwen3.5-4B-Base`; the generator refuses to
load it on any other base model.

Trained on 4 Nov (job 352125). The two adapters side by side:

| | 0.8B adapter | 4B adapter |
| --- | --- | --- |
| Base model | Qwen/Qwen3.5-0.8B-Base | Qwen/Qwen3.5-4B-Base |
| Epochs | 3 | 1 (the 0.8B validation loss was lowest after epoch 1) |
| Precision | fp16 | bf16 |
| Batch per GPU x accumulation | 4 x 4 | 2 x 8 (16 per step either way) |
| Trainable parameters | 10,822,656 (1.42%) | 32,464,896 (0.77%) |
| GPU and training time | RTX 3090, 10,992 s (3.1 h) | NVIDIA L40S, 5,593 s (1.6 h) |
| Validation loss at the end | 3.074 | 2.598 |
| Validation perplexity (reply tokens, every slot filled) | 21.64 | 13.29 |
| Peak GPU memory | not recorded | 14.35 GB |

The two perplexities can be compared directly: the same tokenizer, the same 2,282 validation
examples with every slot filled, and the same `perplexity()` in `src/argubot/eval/metrics.py`. The 4B
adapter is lower than the 0.8B one even at the 0.8B run's best epoch (19.45). Before any training, the
4B base model's loss on the first training batches was already about 2.6 to 2.7 (timing test, job
352080), below the 0.8B adapter's best validation loss, so a large part of the gap is the base model
itself rather than the fine-tuning. The 0.8B trainable count is from the smoke test, which used the
same model and LoRA settings.

A 3-debate check run (job 352127, `outputs/check3_4b`, under 4 minutes for all five systems) gave
fluent, on-topic replies that read like CMV comments, with the bot keeping its side across turns.

## How to reproduce

```
python scripts/download_data.py --all
python scripts/build_cmv_chains.py
python scripts/build_indexes.py                 # about 4 minutes; add --dense on a GPU
python scripts/build_training_data.py --workers 4   # about 30 minutes, about 2.5 GB RAM per worker
python scripts/make_test_debates.py
```
