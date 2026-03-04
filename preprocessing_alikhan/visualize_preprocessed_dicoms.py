import argparse
import io
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np


def list_sample_keys(tar: tarfile.TarFile):
    """Return sorted unique UUID keys that have .frames.npy in the tar."""
    keys = set()
    for m in tar.getmembers():
        name = m.name
        if name.endswith(".frames.npy"):
            keys.add(name[:-len(".frames.npy")])
    return sorted(keys)


def read_npy_from_tar(tar: tarfile.TarFile, member_name: str) -> np.ndarray:
    f = tar.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(member_name)
    data = f.read()
    return np.load(io.BytesIO(data), allow_pickle=False)


def read_json_from_tar(tar: tarfile.TarFile, member_name: str) -> dict:
    f = tar.extractfile(member_name)
    if f is None:
        raise FileNotFoundError(member_name)
    return json.loads(f.read().decode("utf-8"))


def save_mp4(frames_rgb: np.ndarray, out_path: Path, fps: float):
    """Save RGB uint8 frames to MP4 using OpenCV (needs BGR)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    T, H, W, C = frames_rgb.shape
    assert C == 3, frames_rgb.shape

    # mp4v is widely available; if you have h264, you can try 'avc1' or 'H264'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))
    if not vw.isOpened():
        raise RuntimeError(
            "Could not open VideoWriter. Try installing ffmpeg or change codec."
        )

    for f in frames_rgb:
        vw.write(f[..., ::-1])  # RGB -> BGR
    vw.release()


def save_frames(frames_rgb: np.ndarray, out_dir: Path):
    """Save all frames as PNGs into out_dir/frame_000000.png ..."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(frames_rgb):
        p = out_dir / f"frame_{i:06d}.png"
        cv2.imwrite(str(p), f[..., ::-1])  # RGB -> BGR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, help="Path to shard-000000.tar")
    ap.add_argument("--out", required=True, help="Output root directory")
    ap.add_argument("--n", type=int, default=10, help="How many videos to export")
    ap.add_argument("--fps", type=float, default=None,
                    help="Override fps for saved MP4 (default: use metadata target_fps, else 24)")
    args = ap.parse_args()

    out_root = Path(args.out)
    videos_dir = out_root / "videos"
    frames_dir = out_root / "frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.tar, "r") as tar:
        keys = list_sample_keys(tar)
        if not keys:
            raise RuntimeError("No *.frames.npy found in tar")

        export_keys = keys[: min(args.n, len(keys))]
        print(f"Found {len(keys)} samples. Exporting {len(export_keys)}...")

        for k in export_keys:
            npy_name = f"{k}.frames.npy"
            meta_name = f"{k}.metadata.json"

            frames = read_npy_from_tar(tar, npy_name)   # (T,H,W,3) RGB uint8
            meta = {}
            try:
                meta = read_json_from_tar(tar, meta_name)
            except FileNotFoundError:
                pass

            # Determine fps
            fps = args.fps
            if fps is None:
                fps = meta.get("target_fps", None) or meta.get("original_fps", None) or 24.0

            # Save MP4
            mp4_path = videos_dir / f"{k}.mp4"
            save_mp4(frames, mp4_path, fps=fps)

            # Save frames
            sample_frames_dir = frames_dir / k
            save_frames(frames, sample_frames_dir)

            print(f"  Saved {k}: video={mp4_path.name}, frames={len(frames)}")

    print(f"\nDone. Output at: {out_root}")
    print(f"  Videos: {videos_dir}")
    print(f"  Frames: {frames_dir}")


if __name__ == "__main__":
    main()
