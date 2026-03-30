#!/bin/bash
# Setup conda environment for EchoJEPAv2 pretraining
# Run this in a tmux session:
#   tmux new-session -s echojepav2_setup
#   bash /home/ahmedaly/iCardio/EchoJEPAv2/training/setup_env.sh

set -euo pipefail

ENV_NAME="echojepav2"
ECHOJEPEA_DIR="/home/ahmedaly/iCardio/EchoJEPAv2/EchoJEPA"

echo "=== Setting up $ENV_NAME conda environment ==="

# Source conda
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/share/apps/anaconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"

# Remove old env if it exists and is broken
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment $ENV_NAME already exists. Skipping creation."
    echo "To recreate: conda env remove -n $ENV_NAME -y"
else
    echo "Creating conda environment: $ENV_NAME (python=3.11)"
    conda create -n "$ENV_NAME" python=3.11 -y
fi

conda activate "$ENV_NAME"
echo "Activated: $(which python)"

# Install PyTorch with CUDA (adjust cu version if needed)
echo "=== Installing PyTorch ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install EchoJEPA requirements
echo "=== Installing EchoJEPA requirements ==="
pip install \
    tensorboard \
    wandb \
    iopath \
    pyyaml \
    numpy \
    opencv-python \
    submitit \
    braceexpand \
    webdataset \
    timm \
    transformers \
    peft \
    pandas \
    einops \
    beartype \
    psutil \
    h5py \
    fire \
    python-box \
    scikit-image \
    ftfy \
    jupyter

# Install decord (video reader used by EchoJEPA's VideoDataset)
echo "=== Installing decord ==="
pip install decord || {
    echo "WARNING: decord pip install failed, trying conda..."
    conda install -c conda-forge decord -y || echo "WARNING: decord not installed. Only our WebDatasetVideoDataset will work."
}

# Install EchoJEPA as an editable package (makes 'src.*' imports work from anywhere)
echo "=== Installing EchoJEPA package ==="
pip install -e "$ECHOJEPEA_DIR"

# Verify
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'torch:    {torch.__version__}')
print(f'CUDA:     {torch.cuda.is_available()} ({torch.cuda.device_count()} GPUs)')
import numpy; print(f'numpy:    {numpy.__version__}')
import webdataset; print(f'webdataset: OK')
import timm; print(f'timm:     {timm.__version__}')
import sys
sys.path.insert(0, '$ECHOJEPEA_DIR')
from src.datasets.webdataset_video_dataset import WebDatasetVideoDataset
print('WebDatasetVideoDataset: OK')
from src.models.vision_transformer import VisionTransformer
print('VisionTransformer: OK')
print('')
print('=== Setup complete! ===')
print(f'Run training with:')
print(f'  conda activate $ENV_NAME')
print(f'  bash /home/ahmedaly/iCardio/EchoJEPAv2/training/launch_pretrain.sh <shard_dir> <checkpoint_dir> <num_gpus>')
"
