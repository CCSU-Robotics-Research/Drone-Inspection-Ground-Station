"""Tests for YOLO results mapping."""

import numpy as np

from ai.detect import Detections
from ai.yolo import _to_detections

_NAMES = {0: "crack", 1: "spallation"}


class TestToDetections:

    def test_maps_boxes_labels_scores(self):
        det = _to_detections(
            np.array([[10.2, 20.7, 99.5, 120.1],
                      [5.0, 5.0, 50.0, 50.0]]),
            np.array([1.0, 0.0]),
            np.array([0.91, 0.5]),
            _NAMES,
        )
        assert isinstance(det, Detections)
        assert det.boxes == [(10, 21, 100, 120), (5, 5, 50, 50)]
        assert det.labels == ["spallation", "crack"]
        assert det.scores == [0.91, 0.5]

    def test_empty_arrays(self):
        det = _to_detections(
            np.zeros((0, 4)), np.zeros(0), np.zeros(0), _NAMES,
        )
        assert det.boxes == [] and det.labels == []

    def test_heatmap_stays_none(self):
        det = _to_detections(
            np.array([[0, 0, 1, 1]]), np.array([0]),
            np.array([1.0]), _NAMES,
        )
        assert det.heatmap is None
