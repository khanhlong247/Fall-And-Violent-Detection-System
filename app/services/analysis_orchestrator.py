from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

import cv2

from app.core.config import Settings
from app.domain.entities import AnalysisJob, StoredClip
from app.domain.enums import AnalysisStatus, DetectionMode, EventType
from app.domain.schemas import AnalysisResponse, EventClipResponse
from app.infrastructure.google_drive import GoogleDriveStorage
from app.infrastructure.openvino_fall import OpenVinoFallEngine
from app.infrastructure.yolo_violence import YoloViolenceEngine
from app.services.clip_manager import ClipManager
from app.services.cloud_clip_service import CloudClipService
from app.services.event_notifier import EventNotifier
from app.services.event_state_service import EventStateService
from app.services.fall_detector import FallDetector
from app.services.video_source_service import VideoSourceService
from app.services.violence_detector import ViolenceDetector


class AnalysisOrchestrator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.storage = GoogleDriveStorage(settings.google_drive_credentials_json)
        self.video_source_service = VideoSourceService(settings, self.storage)
        self.cloud_clip_service = CloudClipService(settings, self.storage)
        self.notifier = EventNotifier(settings)

    def _build_detectors(self, run_fall_detection: bool, run_violence_detection: bool):
        detectors = []
        if run_fall_detection:
            fall_engine = OpenVinoFallEngine(
                model_xml=self.settings.fall_model_xml,
                device=self.settings.fall_device,
                score_thresh=self.settings.fall_score_thresh,
            )
            detectors.append(FallDetector(fall_engine, tracking="iou", score_thresh=self.settings.fall_score_thresh))

        if run_violence_detection:
            violence_engine = YoloViolenceEngine(
                model_path=self.settings.violence_model_pt,
                conf=self.settings.violence_conf,
                imgsz=self.settings.violence_imgsz,
                device=self.settings.violence_device,
            )
            detectors.append(ViolenceDetector(violence_engine))

        return detectors

    def _build_state_service(self) -> EventStateService:
        return EventStateService(
            trigger_seconds_by_event={
                EventType.FALL: self.settings.fall_trigger_seconds,
                EventType.VIOLENCE: self.settings.violence_trigger_seconds,
            },
            min_positive_frames_by_event={
                EventType.FALL: self.settings.fall_min_positive_frames,
                EventType.VIOLENCE: self.settings.violence_min_positive_frames,
            },
            max_negative_frames_by_event={
                EventType.FALL: self.settings.fall_max_negative_frames,
                EventType.VIOLENCE: self.settings.violence_max_negative_frames,
            },
        )

    def analyze_video_from_cloud(
        self,
        video_name: str,
        run_fall_detection: bool,
        run_violence_detection: bool,
        pre_event_seconds: int,
        post_event_seconds: int,
    ) -> AnalysisResponse:
        local_video_path = self.video_source_service.download_video_from_cloud(video_name)
        return self._analyze_source(
            source=local_video_path,
            source_name=video_name,
            mode=DetectionMode.VIDEO,
            run_fall_detection=run_fall_detection,
            run_violence_detection=run_violence_detection,
            pre_event_seconds=pre_event_seconds,
            post_event_seconds=post_event_seconds,
        )

    def analyze_stream(
        self,
        stream_source: str,
        run_fall_detection: bool,
        run_violence_detection: bool,
        pre_event_seconds: int,
        post_event_seconds: int,
    ) -> AnalysisResponse:
        return self._analyze_source(
            source=stream_source,
            source_name=stream_source,
            mode=DetectionMode.STREAM,
            run_fall_detection=run_fall_detection,
            run_violence_detection=run_violence_detection,
            pre_event_seconds=pre_event_seconds,
            post_event_seconds=post_event_seconds,
        )

    def _analyze_source(
        self,
        source: str | Path,
        source_name: str,
        mode: DetectionMode,
        run_fall_detection: bool,
        run_violence_detection: bool,
        pre_event_seconds: int,
        post_event_seconds: int,
    ) -> AnalysisResponse:
        job = AnalysisJob(
            job_id=uuid4().hex,
            source_name=source_name,
            mode=mode,
            status=AnalysisStatus.RUNNING,
        )

        cap = self.video_source_service.open_capture(source)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = self.settings.default_video_fps_fallback

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        detectors = self._build_detectors(run_fall_detection, run_violence_detection)
        state_service = self._build_state_service()
        clip_manager = ClipManager(
            output_dir=self.settings.event_clips_dir,
            fps=float(fps),
            frame_width=width,
            frame_height=height,
            pre_event_seconds=pre_event_seconds,
            post_event_seconds=post_event_seconds,
            source_name=Path(str(source_name)).stem or "source",
        )

        all_frames_cache: list[tuple[int, any]] = []
        summary = {
            "frames_processed": 0,
            "events_detected": {EventType.FALL.value: 0, EventType.VIOLENCE.value: 0},
            "notifications_sent": {EventType.FALL.value: 0, EventType.VIOLENCE.value: 0},
        }

        frame_index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_seconds = frame_index / fps if fps > 0 else 0.0
                annotated_frame = frame.copy()

                for detector in detectors:
                    if isinstance(detector, FallDetector) and hasattr(detector, "annotate_frame"):
                        annotated_frame, _ = detector.annotate_frame(annotated_frame)

                    if isinstance(detector, ViolenceDetector) and hasattr(detector, "annotate_frame"):
                        annotated_frame, _ = detector.annotate_frame(annotated_frame)

                all_frames_cache.append((frame_index, annotated_frame.copy()))
                clip_manager.push_frame(frame_index, annotated_frame)

                for detector in detectors:
                    records = detector.detect(frame, frame_index, timestamp_seconds)
                    for record in records:
                        entity_id = str(record.metadata.get("track_id", f"{record.event_type.value}_global"))
                        state = state_service.update(source_id=source_name, entity_id=entity_id, record=record)

                        if record.event_detected:
                            clip_manager.register_event(record)
                            summary["events_detected"][record.event_type.value] += 1

                        should_notify, duration = state_service.should_notify(state, timestamp_seconds)
                        if should_notify:
                            self.notifier.notify(
                                source_id=source_name,
                                entity_id=entity_id,
                                record=record,
                                duration_seconds=duration,
                            )
                            state_service.mark_notified(state, timestamp_seconds)
                            summary["notifications_sent"][record.event_type.value] += 1

                for event_type in [EventType.FALL, EventType.VIOLENCE]:
                    clip_manager.maybe_close_window(event_type, frame_index, all_frames_cache)

                frame_index += 1
                summary["frames_processed"] = frame_index
        finally:
            cap.release()

        completed = clip_manager.finalize(frame_index, all_frames_cache)
        uploaded_clips: list[StoredClip] = []
        for clip in completed:
            uploaded_clips.append(self.cloud_clip_service.upload_clip(clip))

        job.status = AnalysisStatus.COMPLETED
        job.finished_at = datetime.utcnow()

        return AnalysisResponse(
            job_id=job.job_id,
            status=job.status,
            mode=job.mode,
            source_name=job.source_name,
            clips=[
                EventClipResponse(
                    event_type=clip.event_type,
                    local_path=str(clip.local_path),
                    remote_file_id=clip.remote_file_id,
                    remote_link=clip.remote_link,
                    start_frame=clip.start_frame,
                    end_frame=clip.end_frame,
                    fps=clip.fps,
                )
                for clip in uploaded_clips
            ],
            summary=summary,
            started_at=job.created_at,
            finished_at=job.finished_at,
        )