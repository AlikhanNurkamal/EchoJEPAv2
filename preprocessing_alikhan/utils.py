import cv2
import numpy as np
from typing import Any


# -------------------------
# FPS resampling
# -------------------------
# Original videos may have different FPS, but we need the sampling rate of 24 FPS for all videos
def resample_fps_nearest(
    video_frames: np.ndarray,
    src_fps: int,
    target_fps: int = 24
) -> np.ndarray:
    """
    Resample video frames to a target FPS using nearest neighbor selection.

    Args:
        video_frames: A numpy array of shape (T, H, W, C) representing the video frames.
        src_fps: The original FPS of the video.
        target_fps: The desired FPS to resample to (default is 24).

    Returns:
        A numpy array of shape (T', H, W, C) where T' is the number of frames after resampling.
    """
    T = len(video_frames)
    duration = T / src_fps  # Total duration of the video in seconds

    T_out = max(1, int(round(duration * target_fps)))  # Number of frames in the resampled video

    # Compute the indices of the source frames to select
    indices = np.round(np.arange(T_out) * (src_fps / target_fps)).astype(int)  # Output ith frame maps to (i * src_fps / target_fps)th source frame

    # Ensure indices are within bounds
    indices = np.clip(indices, 0, T - 1)

    return video_frames[indices]


# -------------------------
# Pixel spacing resampling
# -------------------------
# Original videos may have different pixel spacing, but we need to standardize the pixel spacing for all videos
def standardize_pixel_spacing_single_frame(
    frame: np.ndarray,
    src_spacing_mm: float,
    target_spacing_mm: float,
    interpolation: Any = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Standardize the pixel spacing of a single video frame to a target spacing in mm.

    Args:
        frame: A numpy array of shape (H, W, C) representing a single video frame.
        src_spacing_mm: The original pixel spacing in mm.
        target_spacing_mm: The desired pixel spacing in mm.
        interpolation: The interpolation method to use for resizing (default is cv2.INTER_LANCZOS4).

    Returns:
        A numpy array of shape (H', W', C) where H' and W' are the dimensions of the frame after resampling.
    """
    scale = src_spacing_mm / target_spacing_mm  # Scaling factor for resizing

    # If the scale is is close to 1, we skip resizing
    if abs(scale - 1.0) < 0.01:
        return frame

    original_height, original_width = frame.shape[:2]
    new_height = int(round(original_height * scale))
    new_width = int(round(original_width * scale))

    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=interpolation)
    return resized_frame


def standardize_pixel_spacing_video(video_frames: np.ndarray, src_spacing_mm: float, target_spacing_mm: float, interpolation: Any = cv2.INTER_LANCZOS4) -> np.ndarray:
    """
    Standardize the pixel spacing of all frames in a video to a target spacing in mm.

    Args:
        video_frames: A numpy array of shape (T, H, W, C) representing the video frames.
        src_spacing_mm: The original pixel spacing in mm.
        target_spacing_mm: The desired pixel spacing in mm.
        interpolation: The interpolation method to use for resizing (default is cv2.INTER_LANCZOS4).

    Returns:
        A numpy array of shape (T, H', W', C) where H' and W' are the dimensions of the frames after resampling.
    """
    standardized_frames = []
    for frame in video_frames:
        standardized_frame = standardize_pixel_spacing_single_frame(frame, src_spacing_mm, target_spacing_mm, interpolation)
        standardized_frames.append(standardized_frame)

    return np.array(standardized_frames)
