from enum import Enum


class DetectionMode(str, Enum):
    STREAM = "stream"
    VIDEO = "video"


class EventType(str, Enum):
    FALL = "fall"
    VIOLENCE = "violence"


class AnalysisStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
