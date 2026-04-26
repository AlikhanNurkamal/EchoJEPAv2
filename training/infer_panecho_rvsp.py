#!/usr/bin/env python3
"""
PanEcho RVSP inference on iCardio holdout DICOMs.
Loads preprocessed .npy frames from WebDataset shards,
runs PanEcho RVSP head, computes MAE vs ground-truth labels.

Usage:
  cd ~/Mashrafi/PanEcho
  python ~/iCardio/EchoJEPAv2/training/infer_panecho_rvsp.py --gpu 2
"""
import argparse, io, os, sys, tarfile
import numpy as np
import torch
import cv2
from torchvision.transforms import v2
from torchvision import tv_tensors
from tqdm import tqdm

PANECHO_DIR   = os.path.expanduser('~/Mashrafi/PanEcho')
SHARD_DIRS    = [
    '/hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa',
    '/hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa',
]
HOLDOUT_CSV   = os.path.expanduser(
    '~/iCardio/EchoJEPAv2/training/data_csvs/icardio_rvsp_holdout.csv')
OUT_CSV       = os.path.expanduser(
    '~/iCardio/EchoJEPAv2/training/panecho_rvsp_holdout_preds.csv')

CLIP_LEN  = 16
NUM_CLIPS = 4
MEAN_IN   = [0.485, 0.456, 0.406]
STD_IN    = [0.229, 0.224, 0.225]
# iCardio RVSP normalisation (train set)
RVSP_MEAN = 47.7381
RVSP_STD  = 11.0777


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=2)
    return p.parse_args()


def load_holdout(csv_path):
    """Returns dict: dicom_uuid -> z-score label"""
    labels = {}
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                labels[parts[0]] = float(parts[1])
    return labels


def build_shard_index(shard_dirs, target_uuids):
    """Returns dict: dicom_uuid -> (shard_path, member_name)"""
    import glob
    index = {}
    target_uuids = frozenset(target_uuids)
    for d in shard_dirs:
        for shard in sorted(glob.glob(os.path.join(d, 'shard-*.tar'))):
            with tarfile.open(shard, 'r:') as tar:
                for m in tar.getmembers():
                    if not m.isfile():
                        continue
                    name = os.path.basename(m.name)
                    if not name.endswith('.frames.npy'):
                        continue
                    uuid = name[:-len('.frames.npy')]
                    if uuid in target_uuids and uuid not in index:
                        index[uuid] = shard
    return index


def load_frames(shard_path, uuid):
    """Load (T, H, W, 3) uint8 RGB from shard."""
    with tarfile.open(shard_path, 'r:') as tar:
        for m in tar.getmembers():
            if os.path.basename(m.name) == f'{uuid}.frames.npy':
                f = tar.extractfile(m)
                return np.load(io.BytesIO(f.read()))
    raise FileNotFoundError(uuid)


def sample_clip(frames):
    """Sample a random 16-frame clip, resize 336→256, center-crop to 224."""
    T = frames.shape[0]
    if T < CLIP_LEN:
        reps = -(-CLIP_LEN // T)
        frames = np.tile(frames, (reps, 1, 1, 1))
        T = frames.shape[0]
    start = np.random.randint(0, T - CLIP_LEN + 1)
    clip = frames[start:start + CLIP_LEN]  # (16, H, W, 3)
    # Resize 336→256
    resized = np.stack([
        cv2.resize(f, (256, 256), interpolation=cv2.INTER_AREA)
        for f in clip
    ])  # (16, 256, 256, 3)
    v = tv_tensors.Video(resized.transpose(0, 3, 1, 2))  # (16, 3, 256, 256)
    transform = v2.Compose([
        v2.CenterCrop(224),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=MEAN_IN, std=STD_IN),
    ])
    v = transform(v)                         # (16, 3, 224, 224)
    return v.permute(1, 0, 2, 3)            # (3, 16, 224, 224)


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}')

    # Load PanEcho
    sys.path.insert(0, PANECHO_DIR)
    print('Loading PanEcho...')
    model = torch.hub.load(PANECHO_DIR, 'PanEcho', source='local',
                           pretrained=True, force_reload=False)
    model.to(device).eval()

    # Load holdout labels (z-scores)
    holdout = load_holdout(HOLDOUT_CSV)
    print(f'Holdout DICOMs: {len(holdout)}')

    # Build shard index
    print('Indexing shards...')
    shard_index = build_shard_index(SHARD_DIRS, list(holdout.keys()))
    found = set(shard_index.keys())
    print(f'Found in shards: {len(found)}/{len(holdout)}')

    results = []
    for uuid, z_label in tqdm(holdout.items(), desc='PanEcho RVSP'):
        if uuid not in shard_index:
            continue
        try:
            frames = load_frames(shard_index[uuid], uuid)
            if frames.ndim != 4 or frames.shape[0] == 0:
                continue
            clips = torch.stack([sample_clip(frames) for _ in range(NUM_CLIPS)], 0)
            clips = clips.to(device)
            with torch.inference_mode():
                out = model(clips)
            pred_rvsp = out['RVSP'].squeeze(1).cpu().float().mean().item()
            gt_rvsp   = z_label * RVSP_STD + RVSP_MEAN
            results.append({'dicom_uuid': uuid,
                            'gt_rvsp': gt_rvsp,
                            'pred_rvsp': pred_rvsp})
        except Exception as e:
            print(f'  [skip] {uuid}: {e}')

    import csv
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dicom_uuid', 'gt_rvsp', 'pred_rvsp'])
        w.writeheader()
        w.writerows(results)

    gt   = np.array([r['gt_rvsp']   for r in results])
    pred = np.array([r['pred_rvsp'] for r in results])
    mae  = np.abs(gt - pred).mean()
    rmse = np.sqrt(((gt - pred)**2).mean())
    r2   = 1 - ((gt - pred)**2).sum() / ((gt - gt.mean())**2).sum()
    print(f'\n{"="*50}')
    print(f'  PanEcho RVSP — iCardio holdout (n={len(results)})')
    print(f'  MAE  : {mae:.2f} mmHg')
    print(f'  RMSE : {rmse:.2f} mmHg')
    print(f'  R²   : {r2:.4f}')
    print(f'{"="*50}')
    print(f'Saved: {OUT_CSV}')


if __name__ == '__main__':
    main()
