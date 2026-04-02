from __future__ import annotations

from abc import ABC, abstractmethod

from app.domain.entities import DetectionRecord


class BaseDetector(ABC):
    name: str

    @abstractmethod
    def detect(self, frame, frame_index: int, timestamp_seconds: float) -> list[DetectionRecord]:
        raise NotImplementedError
