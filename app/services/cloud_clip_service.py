from __future__ import annotations

from app.core.config import Settings
from app.domain.entities import StoredClip
from app.infrastructure.google_drive import GoogleDriveStorage
from app.utils.file_utils import normalize_filename


class CloudClipService:
    def __init__(self, settings: Settings, storage: GoogleDriveStorage):
        self.settings = settings
        self.storage = storage

    def upload_clip(self, clip: StoredClip) -> StoredClip:
        remote_name = normalize_filename(clip.local_path.name)
        uploaded = self.storage.upload_file(
            local_path=clip.local_path,
            remote_name=remote_name,
            folder_id=self.settings.google_drive_event_folder_id,
        )
        clip.remote_file_id = uploaded.get("id")
        clip.remote_link = uploaded.get("webViewLink")
        return clip
