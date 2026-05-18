#!/bin/bash
#SBATCH --partition=faculty
#SBATCH --qos=gtqos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=128G
#SBATCH --job-name=echojepav2_eval
#SBATCH -t 12:00:00
#SBATCH --output=training/amd_logs/eval_%j.out
#SBATCH --error=training/amd_logs/eval_%j.err

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
mkdir -p training/amd_logs

# ── Paths ─────────────────────────────────────────────────────────────────────
# Update CHECKPOINT to the path of the trained model you want to evaluate.
# Default: full 100% pretraining run on AMD (200 epochs).
CHECKPOINT="/home/mohammad.yaqub/project/checkpoints/pretrain_icardio_336px_16f/latest.pt"

# Labels directory: copy output_with_labels/output/ to AMD or mount it.
LABELS_DIR="$HOME/project/output_with_labels/output"

# Shard index: rebuild on AMD after pulling, pointing at AMD shard directories.
# Run once:  python evaluation/build_index.py \
#                --shard-dirs /home/mohammad.yaqub/project/preprocessed_by_alikhan_for_echojepa \
#                --output evaluation/shard_index_amd.pkl
SHARD_INDEX="$REPO_DIR/evaluation/shard_index_amd.pkl"

RUN_NAME="100pct_amd_e200"
OUTPUT_DIR="$REPO_DIR/evaluation/results/icardio"

# ── Python path ───────────────────────────────────────────────────────────────
export PYTHONPATH="$REPO_DIR/custom_src:$REPO_DIR/EchoJEPA"

echo "Job ID:    $SLURM_JOB_ID"
echo "Node:      $SLURMD_NODENAME"
echo "GPU:       $ROCR_VISIBLE_DEVICES"
echo "Started:   $(date)"
echo ""

python evaluation/eval_icardio.py \
    --checkpoint "$CHECKPOINT" \
    --run-name   "$RUN_NAME" \
    --labels-dir "$LABELS_DIR" \
    --shard-index "$SHARD_INDEX" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda:0 \
    --batch-size 128 \
    --num-workers 8 \
    --max-dicoms-per-study 3

echo ""
echo "Finished: $(date)"
