#!/usr/bin/env python3
"""
Build the shard index (UUID -> shard_path, byte_offset, size, fmt).

Scans each shard directory one at a time and saves progress after each one,
so if interrupted you won't lose work already done.

Usage:
    python evaluation/build_index.py

Output:
    evaluation/shard_index.pkl       -- the index
    evaluation/shard_index.pkl.dirs  -- tracks which dirs are done
"""

import pickle
from pathlib import Path

from tqdm import tqdm

SHARD_DIRS = [
    "/hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa",
    "/hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa",
    "/hdd1/ahmedaly/preprocessed_valve_eval",
    "/hdd2/ahmedaly/preprocessed_missing_labels",
    "/hdd2/ahmedaly/preprocessed_remaining",
]

INDEX_PATH = Path("evaluation/shard_index.pkl")
META_PATH  = Path("evaluation/shard_index.pkl.dirs")


def scan_tar(tar_path):
    """Return list of (uuid, data_offset, data_size, fmt) for frame files in tar."""
    entries = []
    with open(tar_path, "rb") as f:
        while True:
            hdr_offset = f.tell()
            hdr = f.read(512)
            if len(hdr) < 512 or hdr == b"\x00" * 512:
                break
            name = hdr[:100].rstrip(b"\x00").decode("utf-8", errors="replace")
            size_str = hdr[124:136].rstrip(b"\x00 ").decode("ascii", errors="replace")
            try:
                size = int(size_str, 8) if size_str else 0
            except ValueError:
                break
            data_offset = hdr_offset + 512
            if name.endswith(".frames.npz") or name.endswith(".frames.npy"):
                uuid = name.replace(".frames.npz", "").replace(".frames.npy", "")
                fmt  = "npz" if name.endswith(".frames.npz") else "npy"
                entries.append((uuid, data_offset, size, fmt))
            f.seek(data_offset + ((size + 511) // 512) * 512)
    return entries


def main():
    # Load existing index and completed-dirs tracker
    index = {}
    if INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            index = pickle.load(f)
        print(f"Resuming: {len(index)} entries already indexed")

    done_dirs = set()
    if META_PATH.exists():
        with open(META_PATH, "rb") as f:
            done_dirs = pickle.load(f)

    for shard_dir in SHARD_DIRS:
        shard_dir = Path(shard_dir)
        if str(shard_dir) in done_dirs:
            print(f"[skip] {shard_dir}  (already done)")
            continue

        shards = sorted(shard_dir.glob("shard-*.tar"))
        print(f"\nScanning {len(shards)} shards in {shard_dir} ...")
        for shard_path in tqdm(shards, desc=f"  {shard_dir.name}"):
            try:
                for uuid, offset, size, fmt in scan_tar(shard_path):
                    index[uuid] = (str(shard_path), offset, size, fmt)
            except Exception as e:
                tqdm.write(f"  Warning: skipping {shard_path.name}: {e}")

        # Save progress after each directory
        with open(INDEX_PATH, "wb") as f:
            pickle.dump(index, f)
        done_dirs.add(str(shard_dir))
        with open(META_PATH, "wb") as f:
            pickle.dump(done_dirs, f)
        print(f"  Saved — {len(index)} total entries so far")

    print(f"\nDone. {len(index)} entries → {INDEX_PATH}")


if __name__ == "__main__":
    main()
