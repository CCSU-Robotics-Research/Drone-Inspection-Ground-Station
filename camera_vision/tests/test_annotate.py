"""Tests for the annotator at pixel-level. Synthetic data used"""

import numpy as np
import pytest

from ai.annotate import draw
from ai.detect import Detections

_W, _H = 64, 48


def black_frame():
    return np.zeros((_H, _W, 3), dtype=np.uint8)


def gray_frame(value=100):
    return np.full((_H, _W, 3), value, dtype=np.uint8)


class TestDetections:

    def test_empty_defaults(self):
        d = Detections()
        assert d.boxes == [] and d.labels == [] and d.scores == []
        assert d.heatmap is None

    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            Detections(boxes=[(1, 1, 2, 2)], labels=[], scores=[])


class TestDraw:

    def test_new_array_input(self):
        frame = black_frame()
        out = draw(frame, Detections(
            boxes=[(10, 20, 40, 40)], labels=["x"], scores=[0.5]))
        assert out is not frame
        assert np.array_equal(frame, black_frame())
        assert out.any()

    def test_empty_detections(self):
        frame = gray_frame()
        out = draw(frame, Detections())
        assert out is not frame
        assert np.array_equal(out, frame)

    def test_box_perimeter(self):
        out = draw(black_frame(), Detections(
            boxes=[(10, 20, 40, 40)], labels=["d"], scores=[0.9]))
        assert tuple(out[20, 25]) == (0, 165, 255)
        assert tuple(out[30, 25]) == (0, 0, 0)

    def test_label_renders_above_box(self):
        heat = np.zeros((8, 8), dtype=np.float32)
        heat[:4, :4] = 1.0
        out = draw(gray_frame(), Detections(heatmap=heat))
        assert out[5, 5, 2] > 150 and out[5, 5, 0] < 80
        assert tuple(out[40, 55]) == (100, 100, 100)

    def test_heatmap_below_threshold(self):
        heat = np.full((8, 8), 0.3, dtype=np.float32)
        frame = gray_frame()
        out = draw(frame, Detections(heatmap=heat))
        assert np.array_equal(out, frame)

    def test_coarse_heatmap(self):
        heat = np.array(
            [[1.0, 0.0], [0.0, 0.0]], dtype=np.float32
        )
        out = draw(gray_frame(), Detections(heatmap=heat))
        assert out[4, 4, 2] > 150
        assert tuple(out[_H - 4, _W - 4]) == (100, 100, 100)

    def test_boxes_and_heatmap(self):
        heat = np.zeros((4, 4), dtype=np.float32)
        heat[0, 0] = 1.0
        out = draw(gray_frame(), Detections(
            boxes=[(40, 30, 60, 44)], labels=["spall"],
            scores=[0.8], heatmap=heat))
        assert out[3, 3, 2] > 150
        assert tuple(out[30, 50]) == (0, 165, 255)
