"""Classes for defect detection. Detectors produce a
``Detections`` value that the annotator draws.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

# One instance detection box. [x1, y1, x2, y2] pixel coordinates
Box = Tuple[int, int, int, int]


@dataclass
class Detections:
    """One frame's worth of model output.

    ``boxes``, ``labels`` and ``scores`` are parallel lists
    describing instance detections. ``heatmap`` is an optional
    float32 array with values in [0, 1]; it may be coarser
    than the frame and is resized by the annotator.
    """

    boxes: list = field(default_factory=list)
    labels: list = field(default_factory=list)
    scores: list = field(default_factory=list)
    heatmap: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if not (
            len(self.boxes) == len(self.labels) == len(self.scores)
        ):
            raise ValueError(
                "boxes, labels, and scores must be same length"
            )


class Detector(Protocol):
    """Anything that turns a frame into Detections."""

    def analyze(self, frame: np.ndarray) -> Detections:
        ...


def fuse(heat: Detections, boxes: Detections, size,
         threshold: float = 0.5) -> Detections:
    """Combine DINO's heatmap with YOLO's boxes, contingent upon
    if DINO's heatmap reaches a threshold inside a box."""
    grid = cv2.resize(heat.heatmap.astype(np.float32),
                      (size[1], size[0]),
                      interpolation=cv2.INTER_LINEAR)
    kept = Detections(heatmap=heat.heatmap)
    for box, label, score in zip(boxes.boxes, boxes.labels,
                                 boxes.scores):
        x1, y1, x2, y2 = (max(0, v) for v in box)
        inside = grid[y1:y2, x1:x2]
        if inside.size and inside.max() >= threshold:
            kept.boxes.append(box)
            kept.labels.append(label)
            kept.scores.append(score)
    return kept
