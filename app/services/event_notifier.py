from datetime import datetime
from pathlib import Path
import random
import requests

from app.domain.enums import EventType


class EventNotifier:
    def __init__(self, settings):
        self.settings = settings

    def _to_camera_id(self, source_id: str) -> int:
        stem = Path(source_id).stem.strip()
        if stem.isdigit():
            return int(stem)
        else:
            return 5

    def notify(
        self,
        source_id: str,
        entity_id: str,
        record,
        duration_seconds: float,
        event_clip_file_id: str | None = None,
        event_clip_link: str | None = None,
    ) -> None:
        if record.event_type == EventType.FALL:
            url = self.settings.fall_event_api_url
        elif record.event_type == EventType.VIOLENCE:
            url = self.settings.violence_event_api_url
        else:
            print(f"[ERROR][NOTIFY] Unsupported event type: {record.event_type}")
            return

        camera_id = self._to_camera_id(source_id)

        payload = {
            "camera_id": camera_id,
            "event_type": record.event_type.value,
            "confidence": record.confidence,
            "timestamp": datetime.utcnow().isoformat(),
            "entity_id": entity_id,
            "detector_name": record.detector_name,
            "frame_index": record.frame_index,
            "timestamp_seconds": record.timestamp_seconds,
            "duration_seconds": duration_seconds,
            "metadata": record.metadata,
        }
        
        # Lần 1 gửi url: null, lần 2 gửi url: link video
        payload["url"] = event_clip_link if event_clip_link else None

        if event_clip_link:
            payload["event_clip_link"] = event_clip_link

        if event_clip_file_id:
            payload["event_clip_file_id"] = event_clip_file_id

        print(f"[DEBUG][NOTIFY] POST {url}")
        print(f"[DEBUG][NOTIFY] payload={payload}")

        response = requests.post(
            url,
            json=payload,
            timeout=self.settings.event_api_timeout_seconds,
        )

        print(f"[DEBUG][NOTIFY] response_status={response.status_code}")
        print(f"[DEBUG][NOTIFY] response_text={response.text}")

        response.raise_for_status()