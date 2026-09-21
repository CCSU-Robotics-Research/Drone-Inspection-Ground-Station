"""Unit tests for VideoRecorder."""

import cv2
import numpy as np
import pytest

import record
from record import VideoRecorder

_WIDTH = 64
_HEIGHT = 48


def make_frame(value: int):
    return np.full((_HEIGHT, _WIDTH, 3), value, dtype=np.uint8)


@pytest.fixture
def recorder(tmp_path):
    rec = VideoRecorder(output_dir=tmp_path)
    yield rec
    rec.stop()


class TestVideoRecorder:

    def test_write_and_read_frames(self, recorder):
        path = recorder.start((_WIDTH, _HEIGHT), fps=10.0)
        for i in range(12):
            recorder.write(make_frame(i * 20))
        recorder.stop()

        cap = cv2.VideoCapture(str(path))
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (_HEIGHT, _WIDTH, 3)
            frames += 1
        cap.release()
        assert frames == 12

    def test_lifecycle_flags(self, recorder):
        assert not recorder.is_recording
        recorder.start((_WIDTH, _HEIGHT), fps=10.0)
        assert recorder.is_recording
        recorder.stop()
        assert not recorder.is_recording

    def test_double_start(self, recorder):
        first = recorder.start((_WIDTH, _HEIGHT), fps=10.0)
        second = recorder.start((_WIDTH, _HEIGHT), fps=10.0)
        assert first == second
        recorder.stop()

    def test_stop_without_start(self, recorder):
        recorder.stop()
        recorder.stop()

    def test_write_no_recording_noop(self, recorder):
        recorder.write(make_frame(1))  # No-op

    def test_same_second_recordings(
        self, recorder, monkeypatch
    ):
        class FrozenDateTime:
            @staticmethod
            def now():
                import datetime as real
                return real.datetime(2026, 1, 2, 3, 4, 5)

        monkeypatch.setattr(record, "datetime", FrozenDateTime)

        first = recorder.start((_WIDTH, _HEIGHT), fps=10.0)
        recorder.stop()
        second = recorder.start((_WIDTH, _HEIGHT), fps=10.0)
        recorder.stop()

        assert first != second
        assert first.exists() and second.exists()
        assert second.stem.endswith("_2")

    def test_zero_fps_input(self, recorder):
        path = recorder.start((_WIDTH, _HEIGHT), fps=0.0)
        recorder.write(make_frame(1))
        recorder.stop()

        cap = cv2.VideoCapture(str(path))
        assert cap.get(cv2.CAP_PROP_FPS) == pytest.approx(30.0)
        cap.release()
