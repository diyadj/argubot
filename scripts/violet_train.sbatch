#!/bin/bash
################################################################################
# ArguBot LoRA training on the SCIS Origami GPU cluster (violet.scis.dev).
#
# Headless (non-interactive) job. For the 19 Oct smoke test, prefer running
# notebooks/01_smoke_test.ipynb interactively with `mynotebook` instead -- see
# docs/SCIS_VIOLET.md. Use this script for the full training run later.
#
# Submit with:   sbatch scripts/violet_train.sbatch
# Watch with:    myqueue   /   myjob <jobid>
# Logs appear as <USERNAME>.<JOBID>.out in the directory you ran sbatch from.
################################################################################

## ---- These are the CS425 G3 values (from `myinfo`). Change only if yours differ. ----
#SBATCH --partition=project                    # assigned partition
#SBATCH --account=cs425                        # assigned account
#SBATCH --qos=cs425qos                         # assigned QOS
#SBATCH --mail-user=andreanlhq@gmail.com       # notifications (comma-separate for several)

#SBATCH --job-name=argubot-train
#SBATCH --output=%u.%j.out                     # log file: <user>.<jobid>.out in the submit dir
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mem=32G                              # cs425qos allows up to 100G
#SBATCH --time=08:00:00                        # cs425qos max is 1 day; raise if a run needs it
## Note: cs425qos allows only 1 running / 2 submitted jobs at a time.
## Optional: pin or prefer a GPU model (see docs "Selecting GPUs"), e.g.
## #SBATCH --constraint=a40
## #SBATCH --requeue                           # auto-resubmit if preempted (community queues)

echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"

# --- Environment -------------------------------------------------------------
# Refer to https://violet.scis.dev/docs/Advanced%20settings/module for module names.
module purge
module load Anaconda3/2022.05
eval "$(conda shell.bash hook)"               # do NOT run `conda init` on a shared account

ENV_NAME=argubot
# First run only: create the env and install deps. After one successful run you
# can comment this block out to speed up subsequent submissions.
if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "${ENV_NAME}" python=3.11 -y
    conda activate "${ENV_NAME}"
    pip install -e .
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm
else
    conda activate "${ENV_NAME}"
fi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- Train -------------------------------------------------------------------
# train() in src/argubot/generate/training.py must be implemented first; until then
# this is a wiring check. Swap in scripts/train_lora.py once training is wired.
srun --gres=gpu:1 python3 scripts/train_lora.py --config configs/config.yaml

echo "Job $SLURM_JOB_ID finished at $(date)"
