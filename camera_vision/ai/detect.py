"""Classes for defect detection. Detectors produce a
``Detections`` value that the annotator draws.
"""

import logging
import threading

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


class LiveAnalyzer:
    """Analyzes frames into ``Detections`` via a background thread. The
    ``latest`` frame is drawn and annotated asynchronously, returned with
    ``submit`` to the video feed. The newest frames are always sampled and
    analyzed with the live state maintained"""

    def __init__(self, analyze):
        self._analyze = analyze
        self._frame = None
        self._latest = Detections()
        self._generation = 0
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """Save a copy of the newest frame"""
        with self._lock:
            self._frame = frame.copy()
        self._ready.set()

    @property
    def latest(self) -> Detections:
        with self._lock:
            return self._latest

    def clear(self) -> None:
        """Remove current analysis save states"""
        with self._lock:
            self._frame = None
            self._latest = Detections()
            self._generation += 1

    def stop(self) -> None:
        self._stop.set()
        self._ready.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            self._ready.wait()
            self._ready.clear()
            with self._lock:
                frame, self._frame = self._frame, None
                generation = self._generation
            if frame is None or self._stop.is_set():
                continue
            try:
                result = self._analyze(frame)
            except Exception:
                logging.getLogger(__name__).exception(
                    "live analysis failed"
                )
                continue
            with self._lock:
                if generation == self._generation:
                    self._latest = result
