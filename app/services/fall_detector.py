from __future__ import annotations

import math

import cv2
import numpy as np

from app.domain.entities import DetectionRecord
from app.domain.enums import EventType
from app.infrastructure.openvino_fall import KEYPOINT_DICT, OpenVinoFallEngine
from app.infrastructure.trackers import TrackerIoU, TrackerOKS
from app.services.base_detector import BaseDetector

LINES_BODY = [
    [4, 2], [2, 0], [0, 1], [1, 3],
    [10, 8], [8, 6], [6, 5], [5, 7], [7, 9],
    [6, 12], [12, 11], [11, 5],
    [12, 14], [14, 16], [11, 13], [13, 15],
]


class FallDetector(BaseDetector):
    name = "fall_detector"

    def __init__(self, engine: OpenVinoFallEngine, tracking: str | None = "iou", score_thresh: float = 0.2):
        self.engine = engine
        self.score_thresh = score_thresh
        if tracking == "iou":
            self.tracker = TrackerIoU()
        elif tracking == "oks":
            self.tracker = TrackerOKS()
        else:
            self.tracker = None

    @staticmethod
    def calculate_angle(p1, p2) -> float:
        return abs(math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0])))

    def evaluate_fall(self, body) -> None:
        k = body.keypoints
        s = body.keypoints_score
        idx = KEYPOINT_DICT
        required = [idx["left_shoulder"], idx["right_shoulder"], idx["left_hip"], idx["right_hip"]]

        if not all(s[i] > self.score_thresh for i in required):
            body.is_fall = False
            body.fall_confidence = 0.0
            return

        hip_mid = (k[idx["left_hip"]] + k[idx["right_hip"]]) // 2
        shoulder_mid = (k[idx["left_shoulder"]] + k[idx["right_shoulder"]]) // 2
        angle = self.calculate_angle(hip_mid, shoulder_mid)
        body.fall_angle = float(angle)

        core_kp_conf = float(sum(s[i] for i in required) / len(required))
        angle_score = max(0.0, min(1.0, (45.0 - angle) / 45.0))
        aspect_score = max(0.0, min(1.0, (body.aspect_ratio - 0.8) / 1.2))
        kp_score = max(0.0, min(1.0, (core_kp_conf - self.score_thresh) / max(1e-6, (1.0 - self.score_thresh))))

        body.fall_confidence = max(0.0, min(1.0, 0.50 * angle_score + 0.30 * aspect_score + 0.20 * kp_score))
        body.is_fall = (angle < 60 and body.aspect_ratio > 0.75 and body.fall_confidence >= 0.35)

    def annotate_frame(self, frame):
        """
        Vẽ bbox + skeleton + text lên frame.
        Trả về:
        - annotated_frame
        - bodies (đã infer, track, evaluate_fall)
        """
        annotated = frame.copy()
        bodies = self.engine.infer_bodies(frame)

        if self.tracker:
            bodies = self.tracker.apply(bodies, 0.0)

        for fallback_idx, body in enumerate(bodies):
            if body.track_id < 0:
                body.track_id = fallback_idx + 1

            self.evaluate_fall(body)

            color = (0, 0, 255) if body.is_fall else (0, 255, 0)

            # bbox
            cv2.rectangle(
                annotated,
                (int(body.xmin), int(body.ymin)),
                (int(body.xmax), int(body.ymax)),
                color,
                2,
            )

            # skeleton lines
            lines = []
            for ln in LINES_BODY:
                p1_idx, p2_idx = ln
                if (
                    body.keypoints_score[p1_idx] > self.score_thresh
                    and body.keypoints_score[p2_idx] > self.score_thresh
                ):
                    lines.append(
                        np.array(
                            [body.keypoints[p1_idx], body.keypoints[p2_idx]],
                            dtype=np.int32,
                        )
                    )
            if lines:
                cv2.polylines(annotated, lines, False, color, 2, cv2.LINE_AA)

            # keypoints
            for i, kp in enumerate(body.keypoints):
                if body.keypoints_score[i] > self.score_thresh:
                    cv2.circle(annotated, tuple(map(int, kp)), 3, color, -1)

            # label
            label = (
                f"ID={body.track_id} "
                f"pose={body.score:.2f} "
                f"fall={body.fall_confidence:.2f} "
                f"angle={int(body.fall_angle)}"
            )
            if body.is_fall:
                label = "FALL | " + label

            cv2.putText(
                annotated,
                label,
                (int(body.xmin), max(20, int(body.ymin) - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated, bodies

    def detect(self, frame, frame_index: int, timestamp_seconds: float) -> list[DetectionRecord]:
        bodies = self.engine.infer_bodies(frame)
        if self.tracker:
            bodies = self.tracker.apply(bodies, timestamp_seconds)

        records: list[DetectionRecord] = []
        for fallback_idx, body in enumerate(bodies):
            if body.track_id < 0:
                body.track_id = fallback_idx + 1

            self.evaluate_fall(body)
            records.append(
                DetectionRecord(
                    event_type=EventType.FALL,
                    detector_name=self.name,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    event_detected=body.is_fall,
                    confidence=float(body.fall_confidence),
                    metadata={
                        "track_id": body.track_id,
                        "pose_confidence": float(body.score),
                        "fall_angle": float(body.fall_angle),
                        "bbox": {
                            "xmin": int(body.xmin),
                            "ymin": int(body.ymin),
                            "xmax": int(body.xmax),
                            "ymax": int(body.ymax),
                        },
                    },
                )
            )
        if not records:
            records.append(
                DetectionRecord(
                    event_type=EventType.FALL,
                    detector_name=self.name,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    event_detected=False,
                    confidence=0.0,
                    metadata={"track_id": "fall_global"},
                )
            )
        return records