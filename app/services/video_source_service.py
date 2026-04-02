from __future__ import annotations

from pathlib import Path

import cv2

from app.core.config import Settings
from app.infrastructure.google_drive import GoogleDriveStorage
from app.utils.file_utils import normalize_filename


class VideoSourceService:
    def __init__(self, settings: Settings, storage: GoogleDriveStorage):
        self.settings = settings
        self.storage = storage

    def download_video_from_cloud(self, video_name: str) -> Path:
        safe_name = normalize_filename(video_name)
        destination = self.settings.downloaded_videos_dir / safe_name
        return self.storage.download_file_by_name(
            video_name=video_name,
            destination_path=destination,
            folder_id=self.settings.google_drive_source_folder_id,
        )

    def open_capture(self, source: str | Path):
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            raise ValueError(f"Cannot open source: {source}")
        return cap
