from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.domain.entities import DetectionRecord, StoredClip
from app.domain.enums import EventType
from app.utils.file_utils import build_unique_filename


@dataclass
class ActiveClipWindow:
    event_type: EventType
    source_name: str
    start_frame: int
    last_event_frame: int
    post_padding_frames: int


class ClipManager:
    """
    Cơ chế clip:
    - padding trước: lấy được bao nhiêu thì lấy
    - padding sau: khi không có event mới trong 2 phút sau last_event_frame thì đóng clip
    - fall và violence độc lập
    """

    def __init__(
        self,
        output_dir: Path,
        fps: float,
        frame_width: int,
        frame_height: int,
        pre_event_seconds: int,
        post_event_seconds: int,
        source_name: str,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.pre_event_frames = int(pre_event_seconds * fps)
        self.post_event_frames = int(post_event_seconds * fps)
        self.source_name = source_name

        self.frame_buffer = deque(maxlen=max(1, self.pre_event_frames))
        self.active_windows: dict[EventType, ActiveClipWindow | None] = {
            EventType.FALL: None,
            EventType.VIOLENCE: None,
        }
        self.completed_clips: list[StoredClip] = []

    def push_frame(self, frame_index: int, frame) -> None:
        self.frame_buffer.append((frame_index, frame.copy()))

    def register_event(self, record: DetectionRecord) -> None:
        if not record.event_detected:
            return

        window = self.active_windows[record.event_type]
        if window is None:
            start_frame = max(0, record.frame_index - self.pre_event_frames)
            self.active_windows[record.event_type] = ActiveClipWindow(
                event_type=record.event_type,
                source_name=self.source_name,
                start_frame=start_frame,
                last_event_frame=record.frame_index,
                post_padding_frames=self.post_event_frames,
            )
        else:
            window.last_event_frame = record.frame_index

    def maybe_close_window(self, event_type: EventType, current_frame_index: int, frames_cache: list) -> StoredClip | None:
        window = self.active_windows[event_type]
        if window is None:
            return None

        close_frame = window.last_event_frame + window.post_padding_frames
        if current_frame_index < close_frame:
            return None

        clip = self._write_clip(event_type, window.start_frame, close_frame, frames_cache)
        self.active_windows[event_type] = None
        self.completed_clips.append(clip)
        return clip

    def finalize(self, total_frames: int, frames_cache: list) -> list[StoredClip]:
        for event_type, window in list(self.active_windows.items()):
            if window is None:
                continue
            end_frame = min(total_frames - 1, window.last_event_frame + window.post_padding_frames)
            clip = self._write_clip(event_type, window.start_frame, end_frame, frames_cache)
            self.completed_clips.append(clip)
            self.active_windows[event_type] = None
        return self.completed_clips

    def _write_clip(self, event_type: EventType, start_frame: int, end_frame: int, frames_cache: list) -> StoredClip:
        filename = build_unique_filename(
            prefix=f"{self.source_name}_{event_type.value}_{start_frame}_{end_frame}",
            suffix=".mp4",
        )
        output_path = self.output_dir / filename

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (self.frame_width, self.frame_height),
        )
        for frame_index, frame in frames_cache:
            if start_frame <= frame_index <= end_frame:
                writer.write(frame)
        writer.release()

        return StoredClip(
            event_type=event_type,
            local_path=output_path,
            remote_file_id=None,
            remote_link=None,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=self.fps,
        )
