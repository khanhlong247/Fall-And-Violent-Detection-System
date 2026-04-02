from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import cv2
import numpy as np
from openvino import Core

from app.domain.entities import Body

KEYPOINT_DICT = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
}

Padding = namedtuple("Padding", ["w", "h", "padded_w", "padded_h"])


class OpenVinoFallEngine:
    """
    Giữ logic cốt lõi từ script fall detection:
    - pad/resize
    - infer MoveNet Multipose
    - parse body/keypoints
    """

    def __init__(self, model_xml: Path, device: str = "CPU", score_thresh: float = 0.2):
        self.model_xml = Path(model_xml)
        self.score_thresh = score_thresh
        self.device = device

        if not self.model_xml.exists():
            raise FileNotFoundError(f"Model XML not found: {self.model_xml}")

        self.model_bin = self.model_xml.with_suffix(".bin")
        if not self.model_bin.exists():
            raise FileNotFoundError(f"Model BIN not found: {self.model_bin}")

        self.ie = Core()
        self.pd_model = self.ie.read_model(model=str(self.model_xml))
        self.pd_compiled_model = self.ie.compile_model(model=self.pd_model, device_name=self.device)

        self.pd_input_blob = self.pd_compiled_model.input(0)
        self.pd_output_blob = self.pd_compiled_model.output(0)

        input_shape = list(self.pd_input_blob.shape)
        _, _, self.input_h, self.input_w = input_shape

    def build_padding(self, frame_w: int, frame_h: int) -> Padding:
        if frame_w / frame_h > self.input_w / self.input_h:
            pad_h = int(frame_w * self.input_h / self.input_w - frame_h)
            return Padding(0, pad_h, frame_w, frame_h + pad_h)

        pad_w = int(frame_h * self.input_w / self.input_h - frame_w)
        return Padding(pad_w, 0, frame_w + pad_w, frame_h)

    def preprocess(self, frame, padding: Padding):
        padded = cv2.copyMakeBorder(frame, 0, padding.h, 0, padding.w, cv2.BORDER_CONSTANT)
        resized = cv2.resize(padded, (self.input_w, self.input_h), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32)[None, :]

    def infer_bodies(self, frame) -> list[Body]:
        frame_h, frame_w = frame.shape[:2]
        padding = self.build_padding(frame_w, frame_h)
        frame_nn = self.preprocess(frame, padding)

        inference = self.pd_compiled_model([frame_nn])
        result = np.squeeze(inference[self.pd_output_blob])

        bodies: list[Body] = []
        max_bodies = min(6, len(result))

        for i in range(max_bodies):
            kps = result[i][:51].reshape(17, -1)
            bbox = result[i][51:55].reshape(2, 2)
            score = float(result[i][55])

            if score <= self.score_thresh:
                continue

            ymin, xmin, ymax, xmax = (
                bbox * [padding.padded_h, padding.padded_w]
            ).flatten().astype(int)

            keypoints = kps[:, [1, 0]] * np.array([padding.padded_w, padding.padded_h])
            keypoints = keypoints.astype(int)

            bodies.append(
                Body(
                    score=score,
                    xmin=xmin,
                    ymin=ymin,
                    xmax=xmax,
                    ymax=ymax,
                    keypoints_score=kps[:, 2],
                    keypoints=keypoints,
                    keypoints_norm=keypoints / np.array([frame_w, frame_h]),
                )
            )

        return bodies