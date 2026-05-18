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
#SBATCH --output=/vast/users/mohammad.yaqub/project/EchoJEPAv2/training/amd_logs/eval_%j.out
#SBATCH --error=/vast/users/mohammad.yaqub/project/EchoJEPAv2/training/amd_logs/eval_%j.err

set -euo pipefail

# ── Conda env ─────────────────────────────────────────────────────────────────
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate echojepav2

REPO_DIR="/vast/users/mohammad.yaqub/project/EchoJEPAv2"
cd "$REPO_DIR"

# ── ROCm / MIOpen cache ───────────────────────────────────────────────────────
export MIOPEN_USER_DB_PATH="$HOME/.cache/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$HOME/.cache/miopen_cache"
mkdir -p "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
mkdir -p training/amd_logs

# ── Paths  (all under /vast/users/mohammad.yaqub/project/) ───────────────────
PROJECT="/vast/users/mohammad.yaqub/project"

CHECKPOINT="$PROJECT/checkpoints/pretrain_icardio_336px_16f/latest.pt"

# Labels directory: copy output_with_labels/output/ here or adjust the path.
LABELS_DIR="$PROJECT/output_with_labels/output"

# Shard index: build once with:
#   python evaluation/build_index.py \
#       --shard-dirs $PROJECT/preprocessed_by_alikhan_for_echojepa \
#       --output evaluation/shard_index_amd.pkl
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
    --max-dicoms-per-study 3 \
    --train-fraction 1.0

echo ""
echo "Finished: $(date)"
