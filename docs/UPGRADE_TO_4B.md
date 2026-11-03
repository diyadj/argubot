# Plan: a second run with Qwen3.5-4B-Base

The 0.8B run stays our main result for the presentation on 10 Nov. On top of it we train a
second adapter on Qwen3.5-4B-Base and run the same five systems, so the report can answer one
more question: does conditioning on the claim log matter more when the generator is strong
enough to actually read it? If C1 and B come out level at 0.8B, this run is what tells us
whether the idea failed or the model was just too small.

Everything below runs on the SCIS cluster, logged in as the group account (`CS425G3`), from
`~/argubot`. The group shares **one running job and two submitted jobs** at a time, so the
steps are ordered to keep that one GPU busy without anyone waiting on it.

## What differs between the two runs

Only the model, the precision and the output folders. `configs/config_4b.yaml` builds on
`configs/config.yaml` and changes nothing else, and a test checks that.

| | 0.8B (`configs/config.yaml`) | 4B (`configs/config_4b.yaml`) |
| --- | --- | --- |
| Base model | Qwen/Qwen3.5-0.8B-Base | Qwen/Qwen3.5-4B-Base |
| Training precision | fp16 (it was set up for a Colab T4) | bf16 |
| Generation precision | auto (bf16, what transformers loads by default) | bf16 |
| Adapter | `outputs/adapter` (link to scratch) | `outputs/adapter_4b` |
| Transcripts / results / rating sheets | `outputs/transcripts`, `results`, `human_eval` | the same names with `_4b` |
| Batch (per GPU x accumulation) | 4 x 4 | 2 x 8, the same 16 per step; 4 did not fit next to the 4B weights on a 24 GB card |
| Data, prompts, LoRA r16/alpha32, lr 2e-4, 3 epochs, effective batch 16, warmup 0, decoding, BM25, test debates | same | same |

About warmup: the config used to say 0.03, but transformers 5.x renamed that setting and the
old training code dropped it with only a one-line note, so the 0.8B adapter was trained with no
warmup. The config now says 0.0, which is what really happened, and the 4B run gets the same.
That is now fixed in the code, and any setting transformers does not accept stops the run
instead of disappearing.

## Safety nets you get for free

- **Checkpoints every 100 steps.** If a job dies, the same command picks up from the newest one.
- **A clean stop before the 24 h limit.** Slurm warns the job 20 minutes early; it saves a
  checkpoint, stops, and submits itself again (at most three times).
- **Out of memory?** It retries with half the batch and twice the accumulation, so the effective
  batch stays 16 whether the job lands on a 3090 (24 GB) or an A40/L40S (48 GB). The validation
  passes step down to a smaller batch the same way.
- **Preflight checks.** Every job checks its inputs first and stops in seconds if something is
  missing, instead of after loading a 9 GB model.
- **Nothing gets mixed up.** The 0.8B adapter cannot be loaded on the 4B model or the other way
  round. A finished adapter is never overwritten without `--overwrite`. Checkpoints made with
  different settings are never resumed.

## Once, before anything else (login node)

```bash
ssh CS425G3@10.193.104.102
```

```bash
cd ~/argubot && git status
```

Make sure no job is running (`myqueue`), because a job reads the code as it goes. Then get the
new code. Once the pull request is merged into `main` that is just:

```bash
git pull
```

To try it before the merge, switch the clone to the branch instead (everyone on the group login
then runs this code, which keeps the 0.8B behaviour the same):

```bash
git fetch origin && git checkout claude/gracious-planck-crnbox && git pull
```

Activate the group environment (the same one the batch jobs use):

```bash
module purge && module load Python/3.11.11-GCCcore-13.3.0 && source /common/home/projectgrps/CS425/CS425G3/jupyterlab-venv-pytorch-240/bin/activate
```

Run the tests (about 20 seconds; if pytest is missing, `pip install pytest` first):

```bash
python -m pytest -q
```

Download the 4B model into the group's cache once, so no GPU job waits on it (about 9 GB):

```bash
python scripts/fetch_model.py --config configs/config_4b.yaml
```

Check that training has everything it needs (the GPU line says SKIP on the login node, that is fine):

```bash
python scripts/preflight.py --config configs/config_4b.yaml --stage train
```

## The schedule

| When | GPU job | Meanwhile (no GPU) |
| --- | --- | --- |
| Tue 3 Nov | 4B timing test (about 30 min), then the 0.8B rerun of A to D (queued behind it) | Set up as above |
| Wed 4 Nov | 0.8B evaluation, then start 4B training | Raters start on the 0.8B sheets |
| Thu 5 to Fri 6 Nov | 4B training (the timing test says how long) | Slides with the 0.8B numbers |
| Fri 6 to Sat 7 Nov | 4B check on 3 debates, then the full 4B run of A to D | Read the 4B check transcripts |
| Sun 8 to Mon 9 Nov | 4B evaluation; keep Monday night free | 4B numbers go in as a preview if ready |
| Tue 10 Nov | Nothing queued. Start the demo notebook an hour early | Present |
| 11 to 20 Nov | Anything to rerun | 4B ratings, failure analysis |
| 27 Nov | | Report |

If the timing test says two jobs, everything after it slides by about a day. That still leaves a
week before the report.

## Step 1. The 0.8B rerun (results for the presentation)

The trim fix from 30 Oct means A to D need rerunning on the 0.8B adapter. If that already
happened (for example into `outputs/transcripts_fix`), check it is complete and skip to the evaluation:

```bash
python scripts/preflight.py --config configs/config.yaml --stage eval --transcripts outputs/transcripts_fix
```

Otherwise, after the timing test is submitted (step 2), queue the run behind it:

```bash
sbatch scripts/violet_run.sbatch configs/config.yaml outputs/transcripts_fix
```

Then score it:

```bash
sbatch scripts/violet_eval.sbatch outputs/transcripts_fix outputs/results_fix
```

## Step 2. The 4B timing test (go / no go)

```bash
sbatch --time=01:00:00 scripts/violet_train.sbatch configs/config_4b.yaml --timing-test 30
```

It trains 30 real steps on whatever card it gets, saves nothing, and ends the log with a table
like this (numbers made up):

```
epochs  steps   train h  eval h  total h  jobs
     1   1126      3.13    0.25     3.38     1
     2   2252      6.26    0.38     6.64     1
     3   3378      9.38    0.51     9.89     1

VERDICT: GO: about 9.9 h, fits in one 24 h job.
```

The same report is saved in `outputs/timing/`. What to do with it:

| Verdict | Do this |
| --- | --- |
| GO | Step 3 as it is. |
| GO, but close to the limit | Step 3 as it is. If it runs over, it saves and resubmits itself. |
| GO with resubmits | Step 3 as it is, but it holds the GPU for that many days. If that clashes with the 0.8B work, set `training.num_epochs: 2` in `configs/config_4b.yaml` and say so in the report. |
| NO GO | See "If 4B is too slow" below. |
| A WARNING about the loss not being finite | Stop. Do not start step 3. Send the log over. |
| A WARNING about flash-linear-attention | The batch job tries to install it. If it still says MISSING, run `pip install flash-linear-attention==0.5.2` in the activated venv and repeat the timing test. |

The card matters: a 3090 is slower than an A40 or L40S. If the test lands on one kind and the
full run on another, the time changes. To pin a card, add `--constraint=<feature>` to sbatch
(`sinfo -o "%20N %f"` lists the feature names).

## Step 3. Train the 4B adapter

```bash
sbatch scripts/violet_train.sbatch configs/config_4b.yaml
```

Watch it with `myqueue` and `tail -f CS425G3.<jobid>.out`. Things you will see:

- `Starting from scratch, checkpoints go to outputs/checkpoints_4b`, or `Resuming from ...` on a later job.
- A loss line every 10 steps. It should stay finite and drift down. The 0.8B adapter ended at a
  validation loss of 2.968 after its first epoch (its best) and 3.074 after the third (perplexity 21.6),
  so 4B should land below those.
- An eval loss at the end of each epoch.
- If it gets near 24 h: `Got signal 10: saving a checkpoint ...`, then `Submitted the next job`.
- At the end: the `RECORD FOR REPORT` lines and the validation perplexity.

When it finishes, `outputs/adapter_4b/` holds the adapter and `training_summary.json` (base
model, precision, batch, total time over all jobs, eval loss per epoch, validation perplexity).
The checkpoints are deleted. Keep a copy in scratch next to the 0.8B one:

```bash
S=/common/scratch/projectgrps/CS425/CS425G3/argubot_data && mkdir -p $S/outputs/adapter_4b && cp -r outputs/adapter_4b/. $S/outputs/adapter_4b/ && chmod -R g+rX $S/outputs/adapter_4b
```

```bash
cd $S/outputs/adapter_4b && sha256sum $(ls | grep -v SHA256SUMS) > SHA256SUMS && cd ~/argubot
```

## Step 4. Check on 3 debates before the full run

```bash
sbatch scripts/violet_run.sbatch configs/config_4b.yaml outputs/check3_4b 3
```

Then read the C1 replies:

```bash
python -c "import json; rows=[json.loads(l) for l in open('outputs/check3_4b/C1.jsonl')]; [print(t['bot_text'][:300], '\n---') for r in rows for t in r['turns']]"
```

Look for: replies that stay on the bot's side, use the evidence, and read like CMV replies. If
they are gibberish or empty, stop and send the log over before the full run.

## Step 5. The full 4B run

```bash
sbatch scripts/violet_run.sbatch configs/config_4b.yaml
```

Writes `outputs/transcripts_4b/<system>.jsonl`. If it stops for any reason, submit the same
command again; finished systems are skipped.

## Step 6. Evaluate

```bash
sbatch scripts/violet_eval.sbatch outputs/transcripts_4b outputs/results_4b configs/config_4b.yaml
```

Same five tables as for 0.8B (against A, against B, strict, strict against B, threshold 0.8),
in `outputs/results_4b/eval.log`.

## Step 7. Human ratings for 4B

All five systems, the same as for 0.8B:

```bash
python scripts/make_rating_sheets.py --config configs/config_4b.yaml
```

If the raters are short on time, the core test only (C1 against B):

```bash
python scripts/make_rating_sheets.py --config configs/config_4b.yaml --systems B C1
```

The sheets go to `outputs/human_eval_4b/`, and `python scripts/score_ratings.py --config configs/config_4b.yaml` scores them.

## Step 8 (optional). The demo on 4B

In a `mynotebook` terminal on the GPU node, exactly as in `docs/RUN_ON_VIOLET.md` but with the 4B config:

```bash
python scripts/serve_demo.py --config configs/config_4b.yaml --host 0.0.0.0 --port 8000
```

Replies will be slower than with 0.8B. If the 4B adapter is not ready by 10 Nov, demo the 0.8B one.
In the notebooks, set `CONFIG_PATH = "configs/config_4b.yaml"` in the config cell.

## If 4B is too slow

`configs/config_2b.yaml` is the same setup with Qwen3.5-2B-Base (about half the cost of 4B,
fits a 3090 easily). Every command above works with it; swap the config and the `_4b` folder names
for `_2b`, and fetch the model first:

```bash
python scripts/fetch_model.py --config configs/config_2b.yaml
```

```bash
sbatch --time=01:00:00 scripts/violet_train.sbatch configs/config_2b.yaml --timing-test 30
```

## If something goes wrong

| What you see | What it means and what to do |
| --- | --- |
| `[FAIL] ...` at the top of a job log | Preflight caught it before anything loaded. The line says what to fix. |
| `already holds a trained adapter` | You are about to overwrite an adapter. Check `--config`. Add `--overwrite` only on purpose. |
| `checkpoints ... were made with different settings` | The config changed since the checkpoints were saved. Put it back, or move `outputs/checkpoints_4b` away to start over. |
| `The adapter in ... was trained on ...` | The adapter and the config do not match. Use the config that goes with that adapter. |
| `cannot do bf16` | The job got an old card. Resubmit with `--constraint` for a 3090, A40 or L40S. |
| `Out of GPU memory ... Retrying at 1 x 16` or `Trying again with batch 1` | The card is full. It carries on with a smaller batch. Nothing to do. |
| `Could not resubmit (two jobs already queued?)` | The group already has two jobs queued. When one clears, run the command the log prints. |
| Loss is `nan` | Stop the job (`scancel <jobid>`) and send the log over. |
| A job is stuck pending for hours | Someone else's job (or ours) holds the one slot. `myqueue` shows which. |

## For the report

- Why the second model: the floor-effect argument above, and that the scale comparison changes nothing but the model.
- From `outputs/adapter_4b/training_summary.json`: trainable parameters, training time and GPU, batch per device times accumulation, validation perplexity next to the 0.8B one (21.64; 19.45 after its best epoch, see `docs/DATA_NOTES.md`). Same tokenizer, so the two perplexities are comparable.
- From `outputs/timing/`: seconds per step on each card, if we compare costs.
- The precision difference (fp16 for 0.8B, bf16 for 4B) and the warmup note above.
- Results tables for both models side by side, the C1 against B row being the one that matters.
