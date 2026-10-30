#!/bin/bash
################################################################################
# Score one set of transcripts with every evaluate.py variant, as a batch job on the
# SCIS cluster. Results go to their own folder, so earlier results are never overwritten.
#
#   sbatch scripts/violet_eval.sbatch <transcripts_dir> <results_dir>
#   e.g. sbatch scripts/violet_eval.sbatch outputs/transcripts_fix outputs/results_fix
#
# Writes metrics*.json and flagged*_<system>.jsonl to <results_dir>, and the printed
# tables to <results_dir>/eval.log. Submit from the repo root.
################################################################################

#SBATCH --partition=project
#SBATCH --account=cs425
#SBATCH --qos=cs425qos
#SBATCH --job-name=argubot-eval
#SBATCH --output=%u.%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

TDIR=${1:?usage: sbatch scripts/violet_eval.sbatch <transcripts_dir> <results_dir>}
RDIR=${2:?usage: sbatch scripts/violet_eval.sbatch <transcripts_dir> <results_dir>}
echo "Job $SLURM_JOB_ID on $(hostname): scoring $TDIR into $RDIR, starting at $(date)"

for s in A B C1 C2 D; do
    [ -s "$TDIR/$s.jsonl" ] || { echo "Missing $TDIR/$s.jsonl" >&2; exit 1; }
done

# Same environment as scripts/violet_run.sbatch.
module purge
module load Python/3.11.11-GCCcore-13.3.0
source /common/home/projectgrps/CS425/CS425G3/jupyterlab-venv-pytorch-240/bin/activate

# evaluate.py writes to paths.results_dir, so point a copy of the config at RDIR.
mkdir -p "$RDIR"
sed "s#^\(\s*results_dir:\).*#\1 $RDIR#" configs/config.yaml > "$RDIR/config.yaml"
grep -q "results_dir: $RDIR" "$RDIR/config.yaml" || { echo "Could not set results_dir" >&2; exit 1; }

set -o pipefail
E="python3 scripts/evaluate.py --config $RDIR/config.yaml --transcripts $TDIR --human"
{
    srun $E                              &&
    srun $E --baseline B                 &&
    srun $E --strict                     &&
    srun $E --strict --baseline B        &&
    srun $E --threshold 0.8
} 2>&1 | tee "$RDIR/eval.log" || exit 1

echo "Job $SLURM_JOB_ID finished at $(date)"
