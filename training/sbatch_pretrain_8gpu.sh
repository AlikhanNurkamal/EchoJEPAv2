#!/bin/bash
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=512G
#SBATCH --exclusive
#SBATCH --job-name=echojepav2_pretrain
#SBATCH -t 72:00:00
#SBATCH --output=training/amd_logs/pretrain_%j.out
#SBATCH --error=training/amd_logs/pretrain_%j.err

set -euo pipefail

# ── Conda env ─────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate echojepav2

REPO_DIR="$HOME/project/EchoJEPAv2"
cd "$REPO_DIR"

# ── ROCm / MIOpen cache ───────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="$HOME/.cache/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen_cache"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"

# ── GPU selection ─────────────────────────────────────────────────────────────
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# ── Split filtering ───────────────────────────────────────────────────────────
export ECHOJEPA_ALLOWED_DICOMS="$REPO_DIR/training/train_dicoms.txt"
export ECHOJEPA_HOLDOUT_DICOMS="$REPO_DIR/training/holdout_dicoms.txt"

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH="$REPO_DIR/custom_src:$REPO_DIR/EchoJEPA"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPUs:      $HIP_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo ""

python EchoJEPA/app/main.py \
    --fname training/pretrain_icardio_336px_16f_amd8gpu.yaml \
    --devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7

echo ""
echo "Finished: $(date)"
