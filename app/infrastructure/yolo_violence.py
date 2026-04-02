from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


class YoloViolenceEngine:
    """
    Giữ tinh thần từ script violent detection:
    - load YOLO(best.pt)
    - predict với conf/imgsz/device
    - dùng output boxes của Ultralytics
    """

    def __init__(
        self,
        model_path: Path,
        conf: float = 0.25,
        imgsz: int = 416,
        device: str | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Violence model not found: {self.model_path}")

        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.model = YOLO(str(self.model_path))

    def predict_frame(self, frame):
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            imgsz=self.imgsz,
            save=False,
            device=self.device,
            verbose=False,
        )
        return results[0]

    @property
    def names(self):
        return self.model.names
