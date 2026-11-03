#!/bin/bash
################################################################################
# Train a LoRA adapter as a batch job on the SCIS cluster (violet.scis.dev).
# Submit from the repo root, logged in as the group account.
#
#   Timing test (about 20 min):
#     sbatch --time=01:00:00 scripts/violet_train.sbatch configs/config_4b.yaml --timing-test 30
#   Full training run:
#     sbatch scripts/violet_train.sbatch configs/config_4b.yaml
#
# The config is required, so nobody retrains the 0.8B adapter by accident.
#
# Twenty minutes before the time limit Slurm sends SIGUSR1. Training then saves a checkpoint
# and stops, and this script submits itself again (at most 3 times) to carry on from it.
# Submitting the same command by hand does exactly the same thing.
#
# Watch with:   myqueue      tail -f <USERNAME>.<JOBID>.out
# Mail goes to the address below; override it with sbatch --mail-user=you@smu.edu.sg ...
################################################################################

#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --mail-user=andreanlhq@gmail.com       # change to whoever submits the job
#SBATCH --job-name=argubot-train
#SBATCH --output=%u.%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                              # cs425qos allows up to 100G
#SBATCH --time=24:00:00                        # cs425qos maximum
#SBATCH --signal=USR1@1200                     # warn the training 20 min before the limit
## Note: cs425qos allows only 1 running / 2 submitted jobs at a time for the whole group.

CONFIG=${1:?usage: sbatch scripts/violet_train.sbatch <config.yaml> [--timing-test N] [--overwrite]}
shift
EXTRA=("$@")
RESUBMITS=${ARGUBOT_RESUBMITS:-0}
MAX_RESUBMITS=3

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"
echo "Config $CONFIG ${EXTRA[*]} (resubmission $RESUBMITS of at most $MAX_RESUBMITS)"
[ -f "$CONFIG" ] || { echo "No such config: $CONFIG (submit from the repo root)" >&2; exit 1; }

# --- Environment (the group's Jupyter venv, same as violet_run.sbatch) -----------
VENV=${ARGUBOT_VENV:-/common/home/projectgrps/CS425/CS425G3/jupyterlab-venv-pytorch-240}
if [ ! -x "$VENV/bin/python3" ]; then
    echo "Missing $VENV. Start a PyTorch notebook with mynotebook once to create it." >&2
    exit 1
fi
module purge
module load Python/3.11.11-GCCcore-13.3.0     # the venv's base Python (pyvenv.cfg); provides libpython3.11
source "$VENV/bin/activate"

# The fast Qwen3.5 Gated DeltaNet kernel (about 10x faster). Best effort: training still works without it.
python3 -c "import fla" 2>/dev/null || pip install -q flash-linear-attention==0.5.2 \
    || echo "Could not install flash-linear-attention, so training will be slower."

python3 -c "import torch, argubot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || exit 1

# --- Check everything before spending the GPU -----------------------------------
STAGE=train
OVERWRITE=()
for a in "${EXTRA[@]}"; do
    [ "$a" = "--timing-test" ] && STAGE=timing
    [ "$a" = "--overwrite" ] && OVERWRITE=(--overwrite)
done
python3 scripts/preflight.py --config "$CONFIG" --stage "$STAGE" "${OVERWRITE[@]}" || exit 1

# --- Train ----------------------------------------------------------------------
srun --gres=gpu:1 python3 scripts/train_lora.py --config "$CONFIG" "${EXTRA[@]}"
CODE=$?

if [ "$CODE" -eq 3 ]; then
    echo "Training stopped before the time limit and saved a checkpoint."
    if [ "$RESUBMITS" -lt "$MAX_RESUBMITS" ]; then
        NEXT=$((RESUBMITS + 1))
        if sbatch --export=ALL,ARGUBOT_RESUBMITS=$NEXT scripts/violet_train.sbatch "$CONFIG" "${EXTRA[@]}"; then
            echo "Submitted the next job (resubmission $NEXT of $MAX_RESUBMITS) to carry on."
        else
            echo "Could not resubmit (two jobs already queued?). Carry on by hand with:"
            echo "  sbatch scripts/violet_train.sbatch $CONFIG ${EXTRA[*]}"
        fi
    else
        echo "Already resubmitted $MAX_RESUBMITS times, so stopping here. Read this log, then carry on with:"
        echo "  sbatch scripts/violet_train.sbatch $CONFIG ${EXTRA[*]}"
    fi
    exit 0
fi
if [ "$CODE" -ne 0 ]; then
    echo "Training failed with exit code $CODE at $(date). The newest checkpoint (if any) is kept." >&2
    exit "$CODE"
fi
echo "Job $SLURM_JOB_ID finished at $(date)"
