"""Extract RVSP-labeled DICOMs from sparse shards into dense output shards.

The original WebDataset has ~3,215 shards with ~500K DICOMs total but only
~41K (~0.8%) have RVSP labels. The labeled-dataset loader scans every shard
hunting for the rare labeled samples, making validation extremely slow.

This script:
  1. Loads all RVSP UUIDs (train + val + holdout) from the label CSVs
  2. Scans every input shard in parallel
  3. Extracts only the .frames.npy files matching those UUIDs
  4. Repacks them into new dense tar shards at the output dir

After this, the eval can point dataset_train/val at the dense shard dir
and every batch will hit labeled samples immediately.
"""

import argparse
import io
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob


def load_uuids(csv_paths):
    uuids = set()
    for p in csv_paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                uuids.add(line.split()[0])
    return uuids


def scan_shard(args):
    """Scan one shard, return list of (uuid, npy_bytes) for matching DICOMs."""
    shard_path, target_uuids = args
    out = []
    try:
        with tarfile.open(shard_path, "r:") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                name = os.path.basename(member.name)
                if not name.endswith(".frames.npy"):
                    continue
                uuid = name[: -len(".frames.npy")]
                if uuid not in target_uuids:
                    continue
                f = tar.extractfile(member)
                if f is None:
                    continue
                out.append((uuid, f.read()))
    except Exception as e:
        print(f"[WARN] {shard_path}: {e}", flush=True)
    return shard_path, out


class ShardWriter:
    def __init__(self, out_dir, samples_per_shard=1000):
        self.out_dir = out_dir
        self.samples_per_shard = samples_per_shard
        os.makedirs(out_dir, exist_ok=True)
        self.shard_idx = 0
        self.count_in_shard = 0
        self.total_written = 0
        self.tar = None

    def _open_new_shard(self):
        if self.tar is not None:
            self.tar.close()
        path = os.path.join(self.out_dir, f"shard-{self.shard_idx:06d}.tar")
        self.tar = tarfile.open(path, "w")
        self.count_in_shard = 0

    def write(self, uuid, npy_bytes):
        if self.tar is None:
            self._open_new_shard()
        elif self.count_in_shard >= self.samples_per_shard:
            self.shard_idx += 1
            self._open_new_shard()
        info = tarfile.TarInfo(name=f"{uuid}.frames.npy")
        info.size = len(npy_bytes)
        info.mtime = int(time.time())
        self.tar.addfile(info, io.BytesIO(npy_bytes))
        self.count_in_shard += 1
        self.total_written += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--label-csvs",
        nargs="+",
        default=[
            "/home/ahmedaly/iCardio/EchoJEPAv2/training/data_csvs/icardio_rvsp_train.csv",
            "/home/ahmedaly/iCardio/EchoJEPAv2/training/data_csvs/icardio_rvsp_val.csv",
            "/home/ahmedaly/iCardio/EchoJEPAv2/training/data_csvs/icardio_rvsp_holdout.csv",
        ],
    )
    ap.add_argument(
        "--shard-dirs",
        nargs="+",
        default=[
            "/hdd1/ahmedaly/preprocessed_by_alikhan_for_echojepa",
            "/hdd2/ahmedaly/preprocessed_by_alikhan_for_echojepa",
        ],
    )
    ap.add_argument("--out-dir", default="/data/ahmedaly/rvsp_dense_shards")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--samples-per-shard", type=int, default=1000)
    args = ap.parse_args()

    target_uuids = load_uuids(args.label_csvs)
    print(f"Loaded {len(target_uuids)} target RVSP UUIDs", flush=True)

    shard_paths = []
    for d in args.shard_dirs:
        shard_paths.extend(sorted(glob(os.path.join(d, "shard-*.tar"))))
    print(f"Found {len(shard_paths)} input shards", flush=True)

    writer = ShardWriter(args.out_dir, samples_per_shard=args.samples_per_shard)
    found_uuids = set()

    start = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(scan_shard, (p, target_uuids)) for p in shard_paths]
        for fut in as_completed(futures):
            _, samples = fut.result()
            for uuid, npy_bytes in samples:
                if uuid in found_uuids:
                    continue
                found_uuids.add(uuid)
                writer.write(uuid, npy_bytes)
            completed += 1
            if completed % 50 == 0 or completed == len(shard_paths):
                elapsed = time.time() - start
                rate = completed / max(elapsed, 1)
                eta = (len(shard_paths) - completed) / max(rate, 0.001)
                print(
                    f"[{completed}/{len(shard_paths)}] "
                    f"written={writer.total_written}/{len(target_uuids)} "
                    f"shards_out={writer.shard_idx + 1} "
                    f"elapsed={elapsed:.0f}s eta={eta:.0f}s",
                    flush=True,
                )
    writer.close()
    missing = target_uuids - found_uuids
    print(
        f"Done. Wrote {writer.total_written}/{len(target_uuids)} samples "
        f"across {writer.shard_idx + 1} dense shards to {args.out_dir}. "
        f"Missing: {len(missing)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
