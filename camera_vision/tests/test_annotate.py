"""Tests for the annotator at pixel-level. Synthetic data used"""

import numpy as np
import pytest
import time

from ai.annotate import draw
from ai.detect import Detections, LiveAnalyzer

_W, _H = 64, 48


def black_frame():
    return np.zeros((_H, _W, 3), dtype=np.uint8)


def gray_frame(value=100):
    return np.full((_H, _W, 3), value, dtype=np.uint8)


def small_frame(value):
    return np.full((8, 8, 3), value, dtype=np.uint8)


def wait_until(predicate, timeout=0.3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def tagged(value):
    """Detections that record which frame produced them"""
    return Detections(boxes=[(value, 0, value + 1, 1)], labels=["x"],
                      scores=[1.0])


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
        out = draw(np.zeros((120, 64, 3), dtype=np.uint8), Detections(
            boxes=[(10, 60, 40, 100)], labels=["d"], scores=[0.9]))
        assert tuple(out[100, 25]) == (0, 165, 255)
        assert tuple(out[80, 25]) == (0, 0, 0)

    def test_label_renders_above_box(self):
        out = draw(black_frame(), Detections(
            boxes=[(10, 30, 40, 44)], labels=["crack"],
            scores=[0.9]))
        assert out[:28, :, :].any()

    def test_heatmap_hot_region(self):
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
        assert tuple(out[44, 50]) == (0, 165, 255)


class TestYOLOClassColors:

    def test_known_class_color(self):
        out = draw(black_frame(), Detections(
            boxes=[(10, 20, 40, 40)], labels=["Spallation"],
            scores=[0.9]))
        assert tuple(out[40, 25]) == (255, 140, 0)

    def test_unknown_class_defaults(self):
        out = draw(black_frame(), Detections(
            boxes=[(10, 20, 40, 40)], labels=["mystery"],
            scores=[0.9]))
        assert tuple(out[40, 25]) == (0, 165, 255)


class TestLiveAnalyzer:

    def test_latest_starts_empty(self):
        live = LiveAnalyzer(lambda f: tagged(int(f[0, 0, 0])))
        assert live.latest.boxes == []

    def test_result_appears_after_submit(self):
        live = LiveAnalyzer(lambda f: tagged(int(f[0, 0, 0])))
        live.start()
        live.submit(small_frame(7))
        assert wait_until(lambda: live.latest.boxes != [])
        assert live.latest.boxes[0][0] == 7
        live.stop()

    def test_newest_frame_wins(self):
        seen = []

        def slow(f):
            seen.append(int(f[0, 0, 0]))
            time.sleep(0.2)
            return tagged(seen[-1])

        live = LiveAnalyzer(slow)
        live.start()
        for value in range(1, 6):
            live.submit(small_frame(value))
            time.sleep(0.01)
        assert wait_until(lambda: live.latest.boxes != []
                          and live.latest.boxes[0][0] == 5)
        assert len(seel) < 5
        live.stop()

    def test_clear_discards_live_analysis(self):
        def slow(f):
            time.sleep(0.2)
            return tagged(1)

        live = LiveAnalyzer(slow)
        live.start()
        live.submit(small_frame(1))
        time.sleep(0.05)
        live.clear()
        time.sleep(0.4)
        assert live.latest.boxes == []
        live.stop()

    def test_analysis_error_live(self):
        calls = []

        def flaky(f):
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("model error")
            return tagged(2)

        live = LiveAnalyzer(flaky)
        live.start()
        live.submit(small_frame(1))
        assert wait_until(lambda: len(calls) == 1)
        live.submit(small_frame(2))
        assert wait_until(lambda: live.latest.boxes != [])
        live.stop()

    def test_submit_copies_frame(self):
        seen = []
        live = LiveAnalyzer(
            lambda f: seen.append(int(f[0, 0, 0])) or tagged(0)
        )
        original = small_frame(3)
        live.submit(original)
        original[:] = 9
        live.start()
        assert wait_until(lambda: seen == [3])
        live.stop()

    def test_stop(self):
        live = LiveAnalyzer(lambda f: tagged(0))
        live.stop()
        live.start()
        live.stop()
        live.stop()
