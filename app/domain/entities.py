from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.domain.enums import AnalysisStatus, DetectionMode, EventType


@dataclass
class Body:
    score: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    keypoints_score: np.ndarray
    keypoints: np.ndarray
    keypoints_norm: np.ndarray
    is_fall: bool = False
    fall_angle: float = 0.0
    fall_confidence: float = 0.0
    track_id: int = -1

    @property
    def width(self) -> int:
        return max(1, self.xmax - self.xmin)

    @property
    def height(self) -> int:
        return max(1, self.ymax - self.ymin)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height


@dataclass
class DetectionRecord:
    event_type: EventType
    detector_name: str
    frame_index: int
    timestamp_seconds: float
    event_detected: bool
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventState:
    event_type: EventType
    source_id: str
    entity_id: str
    positive_counter: int = 0
    negative_counter: int = 0
    confirmed_active: bool = False
    active_start_time_seconds: float = 0.0
    last_api_sent_time_seconds: float = 0.0


@dataclass
class StoredClip:
    event_type: EventType
    local_path: Path
    remote_file_id: str | None
    remote_link: str | None
    start_frame: int
    end_frame: int
    fps: float


@dataclass
class AnalysisJob:
    job_id: str
    source_name: str
    mode: DetectionMode
    status: AnalysisStatus = AnalysisStatus.CREATED
    created_at: datetime = field(default_factory=datetime.utcnow)
    finished_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
