#!/bin/bash
################################################################################
# Rehearse the live demo on the SCIS cluster: play the scripted demo debates through
# both models and rank which ones to show (scripts/demo_rehearsal.py).
#
#   sbatch scripts/violet_demo_test.sbatch
#
# 4B first, then 0.8B, about an hour in all. Results print to the log and are saved
# in outputs/demo_rehearsal/. Submit from the repo root, with the queue otherwise empty.
################################################################################

#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --job-name=argubot-demo-test
#SBATCH --output=%u.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date)"

module purge
module load Python/3.11.11-GCCcore-13.3.0
source "${ARGUBOT_VENV:-/common/home/projectgrps/CS425/CS425G3/jupyterlab-venv-pytorch-240}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for CONFIG in configs/config_4b.yaml configs/config.yaml; do
    echo
    echo "################ $CONFIG ################"
    if ! python3 scripts/preflight.py --config "$CONFIG" --stage demo; then
        echo "Skipping $CONFIG: preflight failed." >&2
        continue
    fi
    srun python3 scripts/demo_rehearsal.py --config "$CONFIG"
done

echo "Job $SLURM_JOB_ID finished at $(date)"
