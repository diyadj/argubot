#!/bin/bash
################################################################################
# Run the chat demo as a batch job on the SCIS cluster, instead of inside mynotebook.
# Usually started for you by scripts/start_demo.sh (or scripts/windows/start_demo.ps1).
#
#   sbatch scripts/violet_demo.sbatch [config] [port]
#   defaults: configs/config_4b.yaml, port 8000
#
# The demo runs until the time limit or until you stop it: scancel -n argubot-demo
################################################################################

#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --job-name=argubot-demo
#SBATCH --output=%u.demo.%j.out
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00

CONFIG=${1:-configs/config_4b.yaml}
PORT=${2:-8000}
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
echo "Job $SLURM_JOB_ID on $(hostname) starting at $(date), $CONFIG on port $PORT"

module purge
module load Python/3.11.11-GCCcore-13.3.0
source "${ARGUBOT_VENV:-/common/home/projectgrps/CS425/CS425G3/jupyterlab-venv-pytorch-240}/bin/activate"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 scripts/preflight.py --config "$CONFIG" --stage demo || exit 1
exec python3 -u scripts/serve_demo.py --config "$CONFIG" --host 0.0.0.0 --port "$PORT"
