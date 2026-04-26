"""Convert CAMUS NIfTI half-sequence files to AVI and create train/val/test CSVs."""
import os
import numpy as np
import pandas as pd
import cv2
import nibabel as nib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

CAMUS_ROOT = "/data/ahmedaly/public/CAMUS_public"
AVI_DIR = "/data/ahmedaly/public/CAMUS_public/videos"
SPLIT_CSV = f"{CAMUS_ROOT}/camus_split.csv"
OUT_CSV_DIR = "/home/ahmedaly/iCardio/EchoJEPAv2/training/data_csvs"

TARGET_MEAN = 44.4725
TARGET_STD = 11.7391

os.makedirs(AVI_DIR, exist_ok=True)
os.makedirs(OUT_CSV_DIR, exist_ok=True)


def nifti_to_avi(nii_path: str, avi_path: str, fps: float = 25.0) -> bool:
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()  # (H, W, T) float32
        H, W, T = data.shape

        # Normalize to uint8
        vmin, vmax = data.min(), data.max()
        if vmax > vmin:
            frames_u8 = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            frames_u8 = np.zeros((H, W, T), dtype=np.uint8)

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(avi_path, fourcc, fps, (W, H), isColor=False)
        for t in range(T):
            writer.write(frames_u8[:, :, t])
        writer.release()
        return True
    except Exception as e:
        print(f"ERROR {nii_path}: {e}")
        return False


def main():
    df = pd.read_csv(SPLIT_CSV)

    # Convert all NIfTI files to AVI
    tasks = []
    avi_paths = {}
    for _, row in df.iterrows():
        nii_path = row["path"]
        uid = row["unique_id"]
        avi_path = os.path.join(AVI_DIR, f"{uid}.avi")
        avi_paths[uid] = avi_path
        fps = float(row["FrameRate"]) if not pd.isna(row["FrameRate"]) else 25.0
        tasks.append((nii_path, avi_path, fps))

    print(f"Converting {len(tasks)} NIfTI files to AVI...")
    done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(nifti_to_avi, *t): t[1] for t in tasks}
        for fut in as_completed(futs):
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(tasks)}")

    print("Conversion done. Creating CSVs...")

    # Build CSVs: space-delimited "<avi_path> <z_score_ef>", no header
    for split in ["train", "val", "test"]:
        subset = df[df["split"] == split].copy()
        rows = []
        for _, row in subset.iterrows():
            uid = row["unique_id"]
            avi_path = avi_paths[uid]
            if not os.path.exists(avi_path):
                continue
            z_ef = (row["EF"] - TARGET_MEAN) / TARGET_STD
            rows.append(f"{avi_path} {z_ef:.6f}")
        out_path = os.path.join(OUT_CSV_DIR, f"camus_{split}.csv")
        with open(out_path, "w") as f:
            f.write("\n".join(rows) + "\n")
        print(f"  {split}: {len(rows)} samples → {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
