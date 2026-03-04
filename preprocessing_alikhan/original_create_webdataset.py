import io
import csv
import time
import json
import tarfile
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Set

import cv2
import numpy as np
import pandas as pd

from utils import resample_fps_nearest, standardize_pixel_spacing_video
from utils_fan import get_fan_region, pad_to_square, remove_text_box_in_video


# -----------------------------
# Constants
# -----------------------------
TEXT_BOX_BG = np.array([57, 57, 57])
MIN_RECT_AREA = 2000

STUDY_FIELDS = [
    # Split / demographics
    "study_designation",
    "age_at_visit",
    "height", "height_units",
    "weight", "weight_units",
    "bmi",

    # Primary cardiac label
    "ejection_fraction",

    # Structured pathology labels
    "conditions",
    "characterizations",
    "stratifications",

    # Key measurements (98-99% populated)
    "left_ventricle_diastolic_diameter",
    "left_ventricle_systolic_diameter",
    "left_atrium_dimensions",

    # Per-structure free-text descriptions (88% populated)
    "left_ventricle",
    "right_ventricle",
    "left_atrium",
    "right_atrium",
    "aortic_valve",
    "mitral_valve",
    "tricuspid_valve",
    "pulmonic_valve",
    "pericardium",
    "aortic_root",
    "aortic_arch",
    "pulmonary_artery",

    # Full clinical report
    "conclusions",
]


# -----------------------------
# Resume / progress tracking
# -----------------------------
PROGRESS_FILE = "progress.csv"
PROGRESS_HEADER = ["dicom_uuid", "status", "timestamp"]


def load_done_uuids(progress_path: Path) -> Set[str]:
    """
    Load the set of dicom_uuids that have already been processed (any status)
    from a progress CSV file.  Returns an empty set if the file does not exist.

    Args:
        progress_path (Path): Path to the progress CSV file.

    Returns:
        Set[str]: Set of dicom_uuids that have already been handled.
    """
    if not progress_path.exists():
        return set()
    done: Set[str] = set()
    with open(progress_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add(row["dicom_uuid"])
    return done


def append_progress(progress_path: Path, dicom_uuid: str, status: str):
    """
    Append one record to the progress CSV.  Creates the file with a header
    on first call; subsequent calls append without re-writing the header.

    Args:
        progress_path (Path): Path to the progress CSV file.
        dicom_uuid (str): The DICOM UUID that was processed.
        status (str): Outcome — one of 'processed', 'skipped', 'error'.
    """
    write_header = not progress_path.exists()
    with open(progress_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(PROGRESS_HEADER)
        writer.writerow([dicom_uuid, status, time.strftime("%Y-%m-%dT%H:%M:%S")])


# -----------------------------
# Loading text annotations
# -----------------------------
def load_text_annotations(json_path: Path) -> Tuple[Dict, Dict]:
    """
    Parse the ~/iCardio/preprocessing/json_annotation/combined_updated.json to extract clinical findings and conclusions for each study. Return two fast lookup dictionaries with keys "dicom_uuid" and "study_uuid" respectively.
    
    Args:
        json_path (Path): Path to the combined_updated.json file.
    
    Returns:
        Tuple[Dict, Dict]: Two dictionaries:
            (1) dicom_lookup maps dicom_uuid to its type (Standard, Color)
            (2) study_lookup maps study_uuid to all STUDY_FIELDS
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build lookup dictionaries for DICOMs and studies
    dicom_index = {
        entry["dicom_uuid"]: {
            "dicom_type": entry.get("type")
        }
        for entry in data["dicoms"]
    }
    study_index = {
        entry["study_uuid"]: {
            field: entry.get(field) for field in STUDY_FIELDS
        }
        for entry in data["studies"]
    }

    return dicom_index, study_index


# -----------------------------
# Loading and merging CSVs
# -----------------------------
def load_and_merge_subset_csvs(
    csv_path: Path,
    metadata_csv_path: Path,
    n_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Load the intersected_dicoms_with_metadata{i}.csv and merge with dicom_metadata.csv on dicom_uuid. Take only the subset of rows.

    Args:
        csv_path (Path): Path to the intersected CSV file.
        metadata_csv_path (Path): Path to the dicom_metadata.csv file.
        n_rows (Optional[int]): Number of rows to take from the intersected CSV for processing. If None, process all rows. Default is None.

    Returns:
        pd.DataFrame: Merged DataFrame containing the subset of DICOMs with metadata.
    """
    df = pd.read_csv(csv_path)
    df["dicom_uuid"] = df["dicom_uuid"].astype(str)

    meta = pd.read_csv(metadata_csv_path)
    meta = meta.rename(columns={"icid": "dicom_uuid"})
    meta["dicom_uuid"] = meta["dicom_uuid"].astype(str)

    # Keep relevant columns from metadata
    keep_cols = [
        "dicom_uuid", "manufacturer", "frames_per_second",
        "pixel_height", "pixel_width",
    ]
    # Only keep cols that exist
    keep_cols = [c for c in keep_cols if c in meta.columns]
    meta_sub = meta[keep_cols].drop_duplicates(subset=["dicom_uuid"])

    merged = df.merge(meta_sub, on="dicom_uuid", how="left")
    if n_rows is not None:
        merged = merged.head(n_rows)
    return merged


# -----------------------------
# Processing a single DICOM
# -----------------------------
def load_png_sequence(folder: Path) -> np.ndarray:
    """
    Load a sequence of PNG frames from a folder into a (T, H, W, 3) uint8 array. Sort by filename. Skip unreadable files. Return empty array if no valid frames.

    Args:
        folder (Path): Path to the folder containing PNG frames.

    Returns:
        np.ndarray: Array of shape (T, H, W, 3) with dtype uint8, or empty array if no valid frames.
    """
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".png"
    )

    # Read frames with OpenCV (BGR format)
    frames = [cv2.imread(str(p)) for p in paths]

    # Filter out None frames (failed reads)
    frames = [f for f in frames if f is not None]

    if not frames:
        return np.empty((0,), dtype=np.uint8)

    return np.stack(frames, axis=0)  # Stack into (T, H, W, 3)


def process_one_dicom(
    disk_path: Path,
    src_fps: int,
    target_fps: int,
    src_spacing: float,
    target_spacing: float,
    target_size: int,
) -> np.ndarray | None:
    """
    Process a single DICOM's PNG sequence with the following steps:
        1. Load PNG frames into (T, H, W, 3) uint8 array
        2. Resample pixel spacing to target_spacing (if src_spacing > 0)
        3. Remove text-box from frames
        4. Extract fan region using get_fan_region (with composite reference)
        5. Normalize pixel values to [0, 255] and convert to uint8
        6. Pad to square and resize to (target_size, target_size)
        7. Resample FPS to target_fps (if src_fps > 0)

    Args:
        disk_path (Path): Path to the folder containing PNG frames for this DICOM.
        src_fps (int): Original frames per second of the video. If <= 0, skip FPS resampling.
        target_fps (int): Desired frames per second after resampling.
        src_spacing (float): Original pixel spacing in mm/pixel. If <= 0, skip spacing resampling.
        target_spacing (float): Desired pixel spacing in mm/pixel after resampling.
        target_size (int): Desired output size (height and width) in pixels after resizing.

    Returns:
        np.ndarray | None: Processed video frames as a (T', target_size, target_size, 3) uint8 array, or None if processing failed or no valid frames.
    """
    frames = load_png_sequence(disk_path)
    if len(frames) == 0:
        return None

    # 1. Pixel spacing resample (before any cropping)
    if src_spacing > 0 and target_spacing > 0:
        frames = standardize_pixel_spacing_video(frames, src_spacing, target_spacing)

    # 2. Text-box removal
    frames = remove_text_box_in_video(
        frames.copy(), box_bakcground_pixel=TEXT_BOX_BG, min_rect_area=MIN_RECT_AREA
    )

    # 3. Fan extraction (composite reference from first/middle/last)
    T = len(frames)
    idxs = sorted({0, T // 2, T - 1})
    ref = np.stack([frames[i] for i in idxs], axis=0).max(axis=0).astype(frames.dtype)
    cropped = get_fan_region(ref, threshold=1, video=frames)

    # 4. Normalize
    gmax = float(np.max(cropped)) if np.max(cropped) > 0 else 1.0
    out = []
    for f in cropped:
        norm = (f / gmax * 255).clip(0, 255).astype(np.uint8)
        sq = pad_to_square(norm)
        fin = cv2.resize(sq, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)
        out.append(fin)
    processed = np.stack(out, axis=0)

    # 5. FPS resample
    if src_fps > 0 and target_fps > 0:
        processed = resample_fps_nearest(processed, src_fps, target_fps)

    return processed


# -----------------------------
# Writing WebDataset shards
# -----------------------------
class ShardWriter:
    """Writes WebDataset-style .tar shards."""

    def __init__(self, output_dir: Path, shard_size: int, prefix: str = "shard"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.prefix = prefix

        # Resume: start numbering after the last existing shard so we never
        # overwrite or append to potentially incomplete shards.
        self._shard_idx = self._find_resume_shard_idx(output_dir, prefix)
        self._count = 0
        self._tar: tarfile.TarFile | None = None

    @staticmethod
    def _find_resume_shard_idx(output_dir: Path, prefix: str) -> int:
        """
        Scan *output_dir* for existing shard files matching
        ``<prefix>-NNNNNN.tar`` and return the next index to use.
        Returns 0 when no shards exist yet.
        """
        existing = sorted(output_dir.glob(f"{prefix}-*.tar"))
        if not existing:
            return 0
        # Parse the 6-digit index from the last filename
        last_name = existing[-1].stem          # e.g. "shard-000042"
        try:
            last_idx = int(last_name.split("-")[-1])
        except ValueError:
            last_idx = len(existing) - 1
        return last_idx + 1

    @staticmethod
    def _npy_bytes(arr: np.ndarray) -> bytes:
        """Serialize ndarray to .npy bytes in memory."""
        buf = io.BytesIO()
        np.save(buf, arr)
        return buf.getvalue()

    @staticmethod
    def _add_to_tar(tar: tarfile.TarFile, name: str, data: bytes):
        """Add raw bytes as a file entry in an open tar."""
        info = tarfile.TarInfo(name=name)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    def _open_new_shard(self):
        if self._tar is not None:
            self._tar.close()
        path = self.output_dir / f"{self.prefix}-{self._shard_idx:06d}.tar"
        self._tar = tarfile.open(str(path), "w")
        self._shard_idx += 1
        self._count = 0

    def write_sample(self, key: str, frames: np.ndarray, metadata: dict):
        if self._tar is None or self._count >= self.shard_size:
            self._open_new_shard()

        self._add_to_tar(self._tar, f"{key}.frames.npy", self._npy_bytes(frames))
        self._add_to_tar(self._tar, f"{key}.metadata.json",
                     json.dumps(metadata, indent=2).encode("utf-8"))
        self._count += 1

    def close(self):
        if self._tar is not None:
            self._tar.close()
            self._tar = None


# -----------------------------
# Main function
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Create WebDataset shards.")
    parser.add_argument("--csv", required=True,
                        help="Path to intersected DICOM CSV.")
    parser.add_argument("--metadata-csv", required=True,
                        help="Path to dicom_metadata.csv (has FPS, pixel dims).")
    parser.add_argument("--text-annotations-json",
                        default="/home/ahmedaly/iCardio/preprocessing/json_annotation/combined_updated.json",
                        help="Path to text annotations JSON.")
    parser.add_argument("--num-dicoms", type=int, default=None,
                        help="Process only N DICOMs (for testing).")
    parser.add_argument("--target-fps", type=int, default=24)
    parser.add_argument("--target-spacing", type=float, default=0.36,
                        help="Target mm/pixel.")
    parser.add_argument("--target-size", type=int, default=336,
                        help="Final square image size in pixels.")
    parser.add_argument("--shard-size", type=int, default=1000,
                        help="Max samples per .tar shard.")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for output .tar shards.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)  # Create output directory if it doesn't exist

    # ---- Load and merge CSVs ----
    print("Loading CSVs...")
    df = load_and_merge_subset_csvs(args.csv, args.metadata_csv, n_rows=args.num_dicoms)
    print(f"  Total rows: {len(df):,}")

    # Filter out rows with invalid disk_path
    df = df[df["disk_path"].notna() & df["disk_path"].str.strip().ne("")]
    print(f"  After filtering invalid paths: {len(df):,}")

    # ---- Load text annotations (then save with metadata for each sample) ----
    print(f"Loading text annotations from {args.text_annotations_json}...")
    dicom_lookup, study_lookup = load_text_annotations(Path(args.text_annotations_json))
    print(f"  Loaded {len(dicom_lookup):,} DICOM annotations and {len(study_lookup):,} study annotations")

    # ---- Resume: load already-completed UUIDs ----
    progress_path = output_dir / PROGRESS_FILE
    done_uuids = load_done_uuids(progress_path)
    if done_uuids:
        print(f"Resuming: {len(done_uuids):,} UUIDs already in progress file — skipping them.")
    else:
        print("No previous progress file found — starting from scratch.")

    # ---- Create shard writer ----
    writer = ShardWriter(output_dir, args.shard_size)
    if writer._shard_idx > 0:
        print(f"Resuming shard numbering from index {writer._shard_idx} "
              f"({writer._shard_idx} existing shard(s) kept intact).")

    # ---- Process each DICOM and write to shards ----
    processed = 0
    skipped_problem = 0  # Skipped due to problems (e.g. unreadable/missing frames)
    skipped_already_done = 0  # Skipped because UUID already in progress file
    errors = 0
    t0 = time.time()

    print("\nProcessing DICOMs and writing shards...")
    for row in df.itertuples(index=False):
        dicom_uuid = str(row.dicom_uuid)

        # Skip UUIDs already recorded in the progress file
        if dicom_uuid in done_uuids:
            processed += 1  # Count as processed since we know it's already done, but skip actual work
            skipped_already_done += 1
            continue

        disk_path = Path(str(row.disk_path).strip())
        if not disk_path.is_dir():
            skipped_problem += 1
            append_progress(progress_path, dicom_uuid, "skipped")
            continue

        # Resolve FPS — from metadata CSV or fall back to 0 (skip resampling)
        src_fps = getattr(row, "frames_per_second", 0)
        if pd.isna(src_fps) or src_fps <= 0:
            src_fps = 0

        # Resolve pixel spacing — physical_delta_x is in cm, convert to mm
        raw_delta = getattr(row, "physical_delta_x", 0)
        if pd.isna(raw_delta) or raw_delta <= 0:
            src_spacing = 0
        else:
            src_spacing = float(raw_delta) * 10  # cm -> mm

        try:
            result = process_one_dicom(
                disk_path=disk_path,
                src_fps=src_fps,
                target_fps=args.target_fps,
                src_spacing=src_spacing,
                target_spacing=args.target_spacing,
                target_size=args.target_size,
            )
        except Exception as e:
            errors += 1
            append_progress(progress_path, dicom_uuid, "error")
            if errors <= 10:
                print(f"  [error] {dicom_uuid}: {e}")
            continue

        if result is None or len(result) == 0:
            skipped_problem += 1
            append_progress(progress_path, dicom_uuid, "skipped")
            continue

        # Convert BGR -> RGB for storage
        result_rgb = result[..., ::-1].copy()

        # Lookup text annotations for this DICOM and its study
        study_uuid = str(getattr(row, "study_uuid", ""))
        dicom_info = dicom_lookup.get(dicom_uuid, {})
        study_info = study_lookup.get(study_uuid, {key: None for key in STUDY_FIELDS})

        metadata = {
            # ---- IDs ----
            "dicom_uuid": dicom_uuid,
            "study_uuid": study_uuid,

            # ---- Image metadata ----
            "view": str(getattr(row, "view", "")),
            "dicom_type": dicom_info.get("dicom_type", ""),
            "manufacturer": str(getattr(row, "manufacturer", "")),

            # ---- Original video properties ----
            "original_fps": float(src_fps) if src_fps > 0 else None,
            "original_spacing_mm": src_spacing if src_spacing > 0 else None,
            "n_original_frames": int(getattr(row, "n_frames", 0)),

            # ---- Output video properties ----
            "target_fps": args.target_fps,
            "target_spacing_mm": args.target_spacing,
            "target_size": args.target_size,
            "n_output_frames": len(result_rgb),

            # ---- Study-level annotations ----
            **study_info,
        }

        writer.write_sample(dicom_uuid, result_rgb, metadata)
        append_progress(progress_path, dicom_uuid, "processed")
        processed += 1

        if processed % 100 == 0:
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"  Processed {processed} | skipped {skipped_problem} | "
                  f"errors {errors} | {rate:.1f} samples/s")

    writer.close()
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Processed:                              {processed}")
    print(f"  Skipped (due to problems):              {skipped_problem}")
    print(f"  Skipped (already processed previously): {skipped_already_done}")
    print(f"  Errors:                                 {errors}")
    print(f"  Shards:                                 {writer._shard_idx}")
    print(f"  Output:                                 {output_dir.resolve()}")


if __name__ == "__main__":
    main()
