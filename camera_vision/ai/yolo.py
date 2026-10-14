"""YOLO detector for bounding boxes."""

import numpy as np

from ai.detect import Detections


def _to_detections(xyxy, class_ids, scores, names) -> Detections:
    """Map raw result arrays to a Detections instance"""
    boxes, labels, confs = [], [], []
    for box, cid, score in zip(xyxy, class_ids, scores):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        boxes.append((x1, y1, x2, y2))
        labels.append(str(names[int(cid)]))
        confs.append(float(score))
    return Detections(boxes=boxes, labels=labels, scores=confs)


class YoloDetector:

    def __init__(self, model_path: str = "yolo11n.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)

    def analyze(self, frame: np.ndarray) -> Detections:
        result = self.model(frame, conf=0.4, iou=0.3, verbose=False)[0]
        found = result.boxes
        if found is None or len(found) == 0:
            return Detections()
        return _to_detections(
            found.xyxy.cpu().numpy(),
            found.cls.cpu().numpy(),
            found.conf.cpu().numpy(),
            result.names,
        )
