from pathlib import Path

import cv2


def get_video_metadata(video_path: Path, fps_fallback: float = 25.0) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = fps_fallback

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": float(fps),
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_seconds": duration_seconds,
    }
