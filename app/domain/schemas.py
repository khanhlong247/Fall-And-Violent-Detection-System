from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.domain.enums import AnalysisStatus, DetectionMode, EventType


class VideoAnalysisRequest(BaseModel):
    video_name: str = Field(..., description="Tên video đang lưu trên Google Drive")
    run_fall_detection: bool = True
    run_violence_detection: bool = True
    pre_event_seconds: int = 120
    post_event_seconds: int = 120


class StreamAnalysisRequest(BaseModel):
    stream_source: str = Field(..., description="RTSP/HTTP/webcam path")
    run_fall_detection: bool = True
    run_violence_detection: bool = True
    pre_event_seconds: int = 120
    post_event_seconds: int = 120


class EventClipResponse(BaseModel):
    event_type: EventType
    local_path: str
    remote_file_id: str | None = None
    remote_link: str | None = None
    start_frame: int
    end_frame: int
    fps: float


class AnalysisResponse(BaseModel):
    job_id: str
    status: AnalysisStatus
    mode: DetectionMode
    source_name: str
    clips: list[EventClipResponse] = []
    summary: dict[str, Any] = {}
    started_at: datetime
    finished_at: datetime | None = None
