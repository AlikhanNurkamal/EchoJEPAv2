#!/bin/bash
# VideoMAE ViT-L/16 pretraining on iCardio WebDataset shards.
# Matches JEPA setup: 240 epochs, 224px, 16 frames, holdout denylist applied.
# Single GPU (cuda:0). Batch 16 + accum 8 → effective global batch = 128.
#
# Usage: bash training/run_videomae_pretrain_icardio.sh

set -e

REPO=/home/ahmedaly/iCardio/EchoJEPAv2/EchoJEPA
VMAE_DIR=$REPO/evals/video_classification_frozen/modelcustom/VideoMAE
PYTHON=/home/ahmedaly/.conda/envs/Thesis/bin/python
OUTPUT=/home/ahmedaly/iCardio/checkpoints/pretrain/icardio_vitl16_videomae_224px_16f
HOLDOUT=/home/ahmedaly/iCardio/EchoJEPAv2/training/holdout_dicoms.txt
LOG=/tmp/pretrain_videomae_icardio.log

mkdir -p "$OUTPUT"

export ECHOJEPA_HOLDOUT_DICOMS="$HOLDOUT"
export CUDA_VISIBLE_DEVICES=0

cd "$VMAE_DIR"

nohup $PYTHON run_mae_pretraining_icardio.py \
    --model pretrain_videomae_large_patch16_224 \
    --shard_dirs /hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa \
                 /hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa \
    --output_dir "$OUTPUT" \
    --log_dir "$OUTPUT/tb" \
    --num_frames 16 \
    --input_size 224 \
    --mask_ratio 0.90 \
    --mask_type tube \
    --decoder_depth 4 \
    --batch_size 16 \
    --accum_iter 8 \
    --epochs 240 \
    --warmup_epochs 40 \
    --lr 1.5e-4 \
    --min_lr 1e-5 \
    --warmup_lr 1e-6 \
    --weight_decay 0.05 \
    --save_ckpt_freq 40 \
    --num_workers 6 \
    --seed 234 \
    > "$LOG" 2>&1 &

echo "VideoMAE pretrain launched, PID=$!, log=$LOG"
