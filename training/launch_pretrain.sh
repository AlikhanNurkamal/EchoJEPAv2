#!/bin/bash
# EchoJEPAv2 pretraining launch script
#
# Usage:
#   bash training/launch_pretrain.sh [SHARD_DIR] [CHECKPOINT_DIR] [NUM_GPUS]
#
# Examples:
#   # 4 GPUs, shards in default location
#   bash training/launch_pretrain.sh /path/to/shards /path/to/checkpoints 4
#
#   # 1 GPU (debug / single-node)
#   bash training/launch_pretrain.sh /path/to/shards /path/to/checkpoints 1

set -euo pipefail

# ── Arguments ─────────────────────────────────────────────────────────────────
SHARD_DIR="${1:-/home/ahmedaly/iCardio/preprocessing/output_shards}"
CHECKPOINT_DIR="${2:-/home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_336px_16f}"
NUM_GPUS="${3:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)}"

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
ECHOJEPEA_DIR="$REPO_ROOT/EchoJEPA"
CONFIG="$SCRIPT_DIR/pretrain_icardio_336px_16f.yaml"
PATCHED_CONFIG="/tmp/pretrain_icardio_patched_$$.yaml"

# ── Validate ──────────────────────────────────────────────────────────────────
if [ ! -d "$SHARD_DIR" ]; then
    echo "ERROR: Shard directory not found: $SHARD_DIR"
    exit 1
fi

SHARD_COUNT=$(ls "$SHARD_DIR"/shard-*.tar 2>/dev/null | wc -l)
if [ "$SHARD_COUNT" -eq 0 ]; then
    echo "ERROR: No shard-*.tar files found in $SHARD_DIR"
    exit 1
fi
echo "Found $SHARD_COUNT shards in $SHARD_DIR"

mkdir -p "$CHECKPOINT_DIR"

# ── Patch config with actual paths ────────────────────────────────────────────
# Replace placeholder paths with real ones (avoids editing the committed config)
sed \
    -e "s|/path/to/your/output_shards_dir|$SHARD_DIR|g" \
    -e "s|folder: .*|folder: $CHECKPOINT_DIR|g" \
    "$CONFIG" > "$PATCHED_CONFIG"

echo "Patched config written to: $PATCHED_CONFIG"

# ── Build device list ─────────────────────────────────────────────────────────
DEVICES=""
for i in $(seq 0 $((NUM_GPUS - 1))); do
    DEVICES="$DEVICES cuda:$i"
done
DEVICES="${DEVICES# }"  # trim leading space

echo "Launching with $NUM_GPUS GPU(s): $DEVICES"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo ""

# ── Activate conda env if available ───────────────────────────────────────────
if command -v conda &>/dev/null; then
    # Try to activate the preprocess environment (has torch, cv2, etc.)
    # Change 'preprocess' to your actual conda env name if different
    CONDA_ENV="${CONDA_ENV:-preprocess}"
    source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
    conda activate "$CONDA_ENV" 2>/dev/null || true
fi

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$ECHOJEPEA_DIR"

# Single GPU: use debugmode to avoid multiprocessing overhead
if [ "$NUM_GPUS" -eq 1 ]; then
    echo "Running in single-GPU debug mode..."
    python app/main.py \
        --fname "$PATCHED_CONFIG" \
        --devices cuda:0 \
        --debugmode true
else
    python app/main.py \
        --fname "$PATCHED_CONFIG" \
        --devices $DEVICES
fi

# Cleanup temp config
rm -f "$PATCHED_CONFIG"
