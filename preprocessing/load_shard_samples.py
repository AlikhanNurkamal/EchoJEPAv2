"""
Load 10 video samples (frames + metadata) from a WebDataset shard (.tar file).

Usage:
    python load_shard_samples.py
    python load_shard_samples.py --shard path/to/shard.tar --n-samples 5
"""

import io
import json
import tarfile
import argparse
from pathlib import Path

import numpy as np


def iter_shard_samples(shard_path: Path):
    """
    Iterate over samples in a WebDataset .tar shard.

    Each sample groups consecutive entries by their shared key (the dicom_uuid
    prefix), yielding a dict:
        {
            "key":      str,           # dicom_uuid
            "frames":   np.ndarray,    # (T, H, W, 3) uint8, RGB
            "metadata": dict,          # parsed JSON metadata
        }
    """
    with tarfile.open(str(shard_path), "r") as tar:
        buffer: dict = {}  # key -> {"frames": ..., "metadata": ...}

        for member in tar.getmembers():
            # member name format: <key>.<ext>.npy  or  <key>.metadata.json
            # Split on first dot to get key, remainder is the extension
            name = member.name
            dot = name.index(".")
            key = name[:dot]
            ext = name[dot + 1:]  # e.g. "frames.npy" or "metadata.json"

            if key not in buffer:
                buffer[key] = {}

            raw = tar.extractfile(member).read()

            if ext == "frames.npy":
                buffer[key]["frames"] = np.load(io.BytesIO(raw))
            elif ext == "metadata.json":
                buffer[key]["metadata"] = json.loads(raw.decode("utf-8"))

            # Yield once both parts are available
            if "frames" in buffer[key] and "metadata" in buffer[key]:
                yield {"key": key, **buffer.pop(key)}


def load_n_samples(shard_path: Path, n: int = 10) -> list[dict]:
    """Return the first *n* samples from the shard."""
    samples = []
    for sample in iter_shard_samples(shard_path):
        samples.append(sample)
        if len(samples) >= n:
            break
    return samples


def print_sample_summary(i: int, sample: dict):
    """Print a human-readable summary of one sample."""
    key = sample["key"]
    frames: np.ndarray = sample["frames"]
    meta: dict = sample["metadata"]

    print(f"{'='*60}")
    print(f"Sample {i+1:>2}  |  dicom_uuid: {key}")
    print(f"  Frames shape : {frames.shape}  dtype={frames.dtype}")
    print(f"  FPS (target) : {meta.get('target_fps')}")
    print(f"  Spacing (mm) : {meta.get('target_spacing_mm')}")
    print(f"  Image size   : {meta.get('target_size')}px")
    print(f"  View         : {meta.get('view') or 'N/A'}")
    print(f"  DICOM type   : {meta.get('dicom_type') or 'N/A'}")
    print(f"  Manufacturer : {meta.get('manufacturer') or 'N/A'}")
    print(f"  Study UUID   : {meta.get('study_uuid') or 'N/A'}")

    # Demographic / clinical fields
    print(f"  Age          : {meta.get('age_at_visit') or 'N/A'}")
    print(f"  EF           : {meta.get('ejection_fraction') or 'N/A'}")
    print(f"  Conditions   : {meta.get('conditions') or 'N/A'}")

    conclusions = meta.get("conclusions")
    if conclusions:
        snippet = conclusions[:120].replace("\n", " ")
        print(f"  Conclusions  : {snippet}{'...' if len(conclusions) > 120 else ''}")
    else:
        print(f"  Conclusions  : N/A")


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect N samples from a WebDataset shard."
    )
    parser.add_argument(
        "--shard", type=str, required=True,
        help="Path to the .tar shard file.",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10,
        help="Number of samples to load (default: 10).",
    )
    parser.add_argument(
        "--save-npy", action="store_true",
        help="Save each sample's frames as a .npy file in the current directory.",
    )
    args = parser.parse_args()

    shard_path = Path(args.shard)
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    print(f"Loading {args.n_samples} samples from: {shard_path}\n")
    samples = load_n_samples(shard_path, n=args.n_samples)

    for i, sample in enumerate(samples):
        print_sample_summary(i, sample)

        if args.save_npy:
            out_path = Path(f"{sample['key']}_frames.npy")
            np.save(out_path, sample["frames"])
            print(f"  Saved frames -> {out_path}")

    print(f"\nLoaded {len(samples)} samples successfully.")
    return samples  # convenient when used as a module


if __name__ == "__main__":
    main()
