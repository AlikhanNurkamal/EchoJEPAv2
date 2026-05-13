# src/datasets/embedding_dataset.py
#
# Fast map-style dataset for probe training on pre-extracted embeddings.
# Loads the full embeddings dict into RAM once, then serves (embedding, label)
# pairs directly — no encoder forward pass needed per step.

import os
from logging import getLogger

import numpy as np
import torch

logger = getLogger()


class DummySampler:
    def set_epoch(self, epoch: int):
        pass


def make_embeddingdataset(
    embeddings_path,
    label_csv,
    batch_size,
    num_workers=4,
    pin_mem=True,
    persistent_workers=True,
    world_size=1,
    rank=0,
    drop_last=True,
    training=True,
    **kwargs,
):
    dataset = EmbeddingDataset(
        embeddings_path=embeddings_path,
        label_csv=label_csv,
        world_size=world_size,
        rank=rank,
        training=training,
    )

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        persistent_workers=(num_workers > 0) and persistent_workers,
        shuffle=training,
    )

    data_loader = torch.utils.data.DataLoader(**dl_kwargs)
    data_loader.num_batches = len(data_loader)

    logger.info(
        f"EmbeddingDataset: {len(dataset)} samples, "
        f"~{len(data_loader)} batches/epoch"
    )
    return dataset, data_loader, DummySampler()


class EmbeddingDataset(torch.utils.data.Dataset):
    """
    Map-style dataset over pre-extracted DICOM embeddings.

    embeddings_path : path to .pt file saved by extract_features.py
                      dict[dicom_uuid -> np.float32[D]]
    label_csv       : space-separated file  <dicom_uuid>  <normalised_label>

    Returns: (embedding_tensor [D], label_scalar float32)
    """

    def __init__(
        self,
        embeddings_path: str,
        label_csv: str,
        world_size: int = 1,
        rank: int = 0,
        training: bool = True,
    ):
        super().__init__()
        self.training = training

        # Load embeddings into RAM
        logger.info(f"EmbeddingDataset: loading embeddings from {embeddings_path}")
        raw: dict = torch.load(embeddings_path, map_location="cpu")
        logger.info(f"EmbeddingDataset: loaded {len(raw)} embeddings")

        # Load labels
        label_dict = {}
        with open(label_csv) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uuid, lbl = line.split()
                label_dict[uuid] = float(lbl)
        logger.info(f"EmbeddingDataset: loaded {len(label_dict)} labels from {label_csv}")

        # Intersect: only DICOMs with both an embedding AND a label
        uuids = sorted(set(raw.keys()) & set(label_dict.keys()))
        logger.info(f"EmbeddingDataset: {len(uuids)} samples with both embedding and label")

        # Shard by rank for DDP
        uuids = uuids[rank::max(world_size, 1)]

        self.embeddings = np.stack([raw[u] for u in uuids]).astype(np.float32)  # (N, D)
        self.labels     = np.array([label_dict[u] for u in uuids], dtype=np.float32)
        self.uuids      = uuids

        logger.info(f"EmbeddingDataset (rank {rank}/{world_size}): "
                    f"{len(self.uuids)} samples, embed_dim={self.embeddings.shape[1]}")

    def __len__(self):
        return len(self.uuids)

    def __getitem__(self, idx):
        emb   = torch.from_numpy(self.embeddings[idx])   # (D,)
        label = torch.tensor(self.labels[idx])
        # Wrap in a list to match the (clips, label, clip_indices) interface
        # that run_one_epoch expects: clips is a list of tensors.
        # We add a fake time dimension so the attentive probe sees (1, D).
        return [emb.unsqueeze(0)], label, [np.array([0])]
