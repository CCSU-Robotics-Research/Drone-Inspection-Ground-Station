"""YOLO detector for bounding boxes."""

import numpy as np

from ai.detect import Detections
from pathlib import Path
from ultralytics import YOLO

# WHich trained YOLO size to use: n, s, m, l, x
SIZE = "s"
MODEL = (
    Path(__file__).resolve().parents[1]
    / "models" / f"yolo_codebrim_{SIZE}.pt"
)


class YoloDetector:

    def __init__(self, model_path=MODEL):
        self.model = YOLO(model_path)

    def analyze(self, frame: np.ndarray) -> Detections:
        result = self.model(frame, iou=0.3, verbose=False)[0]
        detections = Detections()

        for box, cid, score in zip(result.boxes.xyxy.tolist(),
                                   result.boxes.cls.tolist(),
                                   result.boxes.conf.tolist()):
            detections.boxes.append(tuple(round(v) for v in box))
            detections.labels.append(result.names[int(cid)])
            detections.scores.append(score)

        return detections
