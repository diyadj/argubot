# Human ratings: how to run them

The proposal promises two group members rating 20 test debates per system, blind to which system
wrote them, on three questions, with Cohen's kappa for agreement. This page is everything needed to
do that for the 0.8B run and the 4B run, start to finish. Nobody needs a GPU for any of it.

## What each rater gets

One CSV file, `rater_1.csv` or `rater_2.csv`. Every row is one debate: three user turns and the
bot's three replies, with no system name anywhere. Both raters get the same rows in a different
order. A third file, `key_PRIVATE.csv`, says which system wrote each row. **Raters never see it.**

| Columns | What goes in them |
| --- | --- |
| `item_id`, `dialogue` | Filled in already. Do not change them. |
| `kept_position` | 0 or 1 |
| `answered_counterargument` | 0 or 1 |
| `used_support_correctly` | 0 or 1 |
| `notes` | Optional, anything worth remembering for the failure analysis |

All five systems is 100 rows per rater (about 2 hours). Only B and C1, the core test, is 40 rows
(about 45 minutes).

## First: agree the rules

Both raters must judge the same way, and the 4B sheets must be judged the same way as the 0.8B
ones. If the 0.8B ratings were already done with agreed rules, use those. Otherwise this is a
suggestion to agree on before anyone starts:

- **kept_position.** 1 if the bot argues the same side in every reply. A concession that keeps the
  side ("uniforms can be cheaper, but they still...") counts as kept. 0 if it switches side, or a
  reply directly contradicts an earlier one.
- **answered_counterargument.** 1 if at least two of the three replies respond to the point the user
  just made. 0 if the replies mostly ignore it, change the subject, or would fit any message.
- **used_support_correctly.** 1 if the reasons, examples or facts the bot gives actually back up its
  side and stay on topic. 0 if it gives no support, the support is off topic, or it argues for the
  other side. Judge from the dialogue only; the sheet does not show the retrieved passages.

Rate on your own and do not compare answers until both sheets are finished. Agreement is the
thing being measured.

## 1. Make the sheets (on the cluster)

Log in as the group account, then:

```bash
cd ~/argubot && module purge && module load Python/3.11.11-GCCcore-13.3.0 && source /common/home/projectgrps/CS425/CS425G3/jupyterlab-venv-pytorch-240/bin/activate
```

**4B**, once `outputs/transcripts_4b` has all five systems:

```bash
python scripts/make_rating_sheets.py --config configs/config_4b.yaml
```

or, for the core test only:

```bash
python scripts/make_rating_sheets.py --config configs/config_4b.yaml --systems B C1
```

The sheets go to `outputs/human_eval_4b/`.

**0.8B.** The sheets have to come from `outputs/transcripts_fix`, the run after the 30 Oct fix,
because that is what the reported numbers come from. Check what is already in `outputs/human_eval`:

```bash
ls -la --time-style=long-iso outputs/human_eval/ && ls -ld --time-style=long-iso outputs/transcripts_fix
```

If `outputs/human_eval` holds sheets older than `transcripts_fix`, they were made from the old
transcripts. See whether anyone has started on them before doing anything:

```bash
python -c "import csv; rows=list(csv.DictReader(open('outputs/human_eval/rater_1.csv', encoding='utf-8'))); print(sum(1 for r in rows if r['kept_position'].strip()), 'of', len(rows), 'rated')"
```

Then make fresh sheets from the right transcripts in their own folder (nothing old is touched):

```bash
python scripts/make_rating_sheets.py --config configs/config.yaml --transcripts outputs/transcripts_fix --out outputs/human_eval_fix
```

The script never writes over existing sheets, so a second run by mistake cannot wipe ratings.

## 2. Get the sheets to the raters

From your laptop (PowerShell, SMU VPN on if you are off campus), copy one file per rater:

```bash
scp CS425G3@10.193.104.102:~/argubot/outputs/human_eval_4b/rater_1.csv .
```

```bash
scp CS425G3@10.193.104.102:~/argubot/outputs/human_eval_4b/rater_2.csv .
```

Send each rater only their own file. Never send `key_PRIVATE.csv`.

## 3. Rating

Open the CSV in Google Sheets (File, Import) or Excel. Fill the three columns with 0 or 1 for every
row. The `dialogue` cells are long; widen the column and turn on text wrapping. When done, save it
as CSV again with the same name: in Google Sheets, File, Download, Comma-separated values; in
Excel, Save As, "CSV UTF-8". Leave the header row and `item_id` exactly as they were.

## 4. Bring the sheets back

From the laptop, put each finished sheet back where it came from, with the same name:

```bash
scp rater_1.csv CS425G3@10.193.104.102:~/argubot/outputs/human_eval_4b/rater_1.csv
```

```bash
scp rater_2.csv CS425G3@10.193.104.102:~/argubot/outputs/human_eval_4b/rater_2.csv
```

## 5. Score

On the cluster, with the environment on as in step 1:

```bash
python scripts/score_ratings.py --config configs/config_4b.yaml
```

For the 0.8B sheets made from `transcripts_fix`:

```bash
python scripts/score_ratings.py --config configs/config.yaml --dir outputs/human_eval_fix --results outputs/results_fix
```

It first checks both sheets and lists any blank cell or value that is not 0 or 1, and stops until
they are fixed. Then it prints Cohen's kappa for each question and each system's average score,
and saves both to `human_eval.json` in the results folder (`outputs/results_4b` for 4B).

## If something goes wrong

| What you see | What to do |
| --- | --- |
| `already has [...] They may hold ratings` | Sheets are already there. Use `--out` for a new folder, or `--overwrite` only if you are sure they are blank. |
| `rater_1.csv: N problem(s)` when scoring | Blank cells or values other than 0 or 1. Fix them in the sheet, copy it back, score again. |
| Strange characters in the dialogue after saving | The file was saved in the wrong encoding. Save again as "CSV UTF-8". |
| `No such file or directory: ... C2.jsonl` | Making all five systems' sheets before the run finished. Check with `python scripts/preflight.py --config configs/config_4b.yaml --stage eval`. |
| `No module named 'argubot'` | The environment is not on. Run the `module purge && ...` line from step 1. |
