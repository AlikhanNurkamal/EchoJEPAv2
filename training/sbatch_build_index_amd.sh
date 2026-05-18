#!/bin/bash
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=32G
#SBATCH --job-name=build_shard_index
#SBATCH -t 03:00:00
#SBATCH --output=training/amd_logs/build_index_%j.out
#SBATCH --error=training/amd_logs/build_index_%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate echojepav2

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

mkdir -p training/amd_logs

echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURMD_NODENAME"
echo "Started: $(date)"
echo ""

python evaluation/build_index.py \
    --shard-dirs /vast/users/mohammad.yaqub/project/preprocessed_by_alikhan_for_echojepa \
    --output evaluation/shard_index_amd.pkl

echo ""
echo "Finished: $(date)"
