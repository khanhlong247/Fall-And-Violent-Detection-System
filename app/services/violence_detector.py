from __future__ import annotations

import cv2

from app.domain.entities import DetectionRecord
from app.domain.enums import EventType
from app.infrastructure.yolo_violence import YoloViolenceEngine
from app.services.base_detector import BaseDetector


class ViolenceDetector(BaseDetector):
    name = "violence_detector"

    def __init__(self, engine: YoloViolenceEngine):
        self.engine = engine

    def annotate_frame(self, frame):
        """
        Vẽ bbox + class name + confidence + label VIOLENCE lên frame.
        Trả về:
        - annotated_frame
        - result YOLO
        """
        annotated = frame.copy()
        result = self.engine.predict_frame(frame)
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return annotated, result

        names = self.engine.names

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [int(x) for x in box.xyxy[0].tolist()]
            x1, y1, x2, y2 = xyxy

            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            class_name_str = str(class_name)
            is_violence = class_name_str.lower() in {"violence", "violent", "fight", "fighting"}

            color = (0, 0, 255) if is_violence else (0, 255, 0)

            # bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # label
            label = f"{class_name_str} {conf:.2f}"
            if is_violence:
                label = f"VIOLENCE | {label}"

            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated, result

    def detect(self, frame, frame_index: int, timestamp_seconds: float) -> list[DetectionRecord]:
        result = self.engine.predict_frame(frame)
        boxes = result.boxes

        records: list[DetectionRecord] = []
        if boxes is None or len(boxes) == 0:
            records.append(
                DetectionRecord(
                    event_type=EventType.VIOLENCE,
                    detector_name=self.name,
                    frame_index=frame_index,
                    timestamp_seconds=timestamp_seconds,
                    event_detected=False,
                    confidence=0.0,
                    metadata={"track_id": "violence_global", "detections": []},
                )
            )
            return records

        detections = []
        best_conf = 0.0
        event_detected = False

        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [round(x, 2) for x in box.xyxy[0].tolist()]
            names = self.engine.names
            class_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]

            detections.append(
                {
                    "index": idx,
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": xyxy,
                }
            )
            best_conf = max(best_conf, conf)
            if str(class_name).lower() in {"violence", "violent", "fight", "fighting"}:
                event_detected = True

        if not event_detected:
            event_detected = len(detections) > 0

        records.append(
            DetectionRecord(
                event_type=EventType.VIOLENCE,
                detector_name=self.name,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                event_detected=event_detected,
                confidence=float(best_conf),
                metadata={"track_id": "violence_global", "detections": detections},
            )
        )
        return records