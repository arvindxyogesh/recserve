#!/bin/bash
# Generic Great Lakes job: runs `python -m <module> [args]` in the repo's venv.
# Partition, GPUs and account are chosen at submit time, e.g.
#   sbatch -p spgpu --gpus=1 scripts/slurm/run.sbatch recserve.retrieval.train_two_tower
#SBATCH --job-name=recserve
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
module load python/3.11.5
source .venv/bin/activate
echo "host=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv || true
python -m "$@"
