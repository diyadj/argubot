#!/bin/bash
################################################################################
# Run systems A, B, C1, C2 and D on the test debates as a batch job on the SCIS
# cluster (violet.scis.dev), so nobody has to stay logged in.
#
# Each system runs as its own step and writes <out>/<system>.jsonl when it finishes.
# A system whose file already exists is skipped, so if the job dies or hits the time
# limit, just submit it again and it carries on from the next system. Delete a file
# to rerun that system.
#
# 3-debate check (step 3):   sbatch scripts/violet_run.sbatch outputs/check3 3
# Full run (step 4):         sbatch scripts/violet_run.sbatch
# Watch with:                myqueue   /   tail -f <USERNAME>.<JOBID>.out
#
# Needs the adapter, passages and BM25 index linked in from scratch first
# (docs/DATA_NOTES.md, "Built files on the SCIS cluster" and "Trained LoRA adapter").
# Submit from the repo root.
################################################################################

## ---- These are the CS425 G3 values (from `myinfo`). Change only if yours differ. ----
#SBATCH --partition=project                    # assigned partition
#SBATCH --account=cs425                        # assigned account
#SBATCH --qos=cs425qos                         # assigned QOS
#SBATCH --mail-user=andreanlhq@gmail.com       # change to whoever submits the job

#SBATCH --job-name=argubot-run
#SBATCH --output=%u.%j.out                     # log file: <user>.<jobid>.out in the submit dir
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1                           # for the whole job, so each srun step below can use it
#SBATCH --mem=32G                              # the BM25 index and 1.2M passages need ~10 GB
#SBATCH --time=20:00:00                        # cs425qos max is 1 day; resubmit to continue
## Note: cs425qos allows only 1 running / 2 submitted jobs at a time.

OUT=${1:-outputs/transcripts}
LIMIT=${2:-}
SYSTEMS="A B C1 C2 D"

echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"
echo "Writing to $OUT${LIMIT:+, first $LIMIT debates only}"

for f in outputs/adapter data/processed/passages.jsonl data/indexes/bm25_docs.npy; do
    if [ ! -e "$f" ]; then
        echo "Missing $f. Link it in from scratch first (docs/DATA_NOTES.md)." >&2
        exit 1
    fi
done

# --- Environment ---------------------------------------------------------------
# The group's Jupyter venv, which ran the five systems and the evaluation (requirements.txt is
# pinned to it). The Anaconda3/2022.05 module this used to load no longer exists on the cluster.
VENV=/common/home/projectgrps/CS425/CS425G3/jupyterlab-venv-pytorch-240
if [ ! -x "$VENV/bin/python3" ]; then
    echo "Missing $VENV. Start a PyTorch notebook with mynotebook once to create it." >&2
    exit 1
fi
module purge
module load Python/3.11.11-GCCcore-13.3.0     # the venv's base Python (pyvenv.cfg); provides libpython3.11
source "$VENV/bin/activate"

python3 -c "import torch, argubot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

# --- Run, one system at a time -----------------------------------------------
for S in $SYSTEMS; do
    if [ -s "$OUT/$S.jsonl" ]; then
        echo "System $S: $OUT/$S.jsonl already exists, skipping"
        continue
    fi
    echo "System $S starting at $(date)"
    srun --gres=gpu:1 python3 scripts/run_systems.py --systems "$S" --out "$OUT" ${LIMIT:+--limit "$LIMIT"} || exit 1
done

echo "Job $SLURM_JOB_ID finished at $(date)"
