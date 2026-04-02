from __future__ import annotations

import requests

from app.core.config import Settings
from app.domain.entities import DetectionRecord
from app.domain.enums import EventType
from app.utils.time_utils import iso_utc_now


class EventNotifier:
    def __init__(self, settings: Settings):
        self.settings = settings

    def notify(self, source_id: str, entity_id: str, record: DetectionRecord, duration_seconds: float) -> None:
        url = self.settings.fall_event_api_url if record.event_type == EventType.FALL else self.settings.violence_event_api_url
        payload = {
            "source_id": source_id,
            "entity_id": entity_id,
            "event_type": record.event_type.value,
            "confidence": float(record.confidence),
            "duration_seconds": float(duration_seconds),
            "frame_index": record.frame_index,
            "timestamp": iso_utc_now(),
            "metadata": record.metadata,
        }
        try:
            requests.post(url, json=payload, timeout=self.settings.event_api_timeout_seconds)
        except Exception as exc:
            print(f"[WARN] Failed to notify {record.event_type.value}: {exc}")
