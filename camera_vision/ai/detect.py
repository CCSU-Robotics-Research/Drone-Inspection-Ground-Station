"""Classes for defect detection. Detectors produce a
``Detections`` value that the annotator draws.
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple

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
        if not (len(self.boxes) == len(self.labels)
            == len(self.scores)):
            raise ValueError(
                "boxes, labels, and sores must be same length"
            )

