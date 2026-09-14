"""Unit tests for the camera feed reader."""

import cv2
import os
import numpy as np
import pytest

from read_feed import CameraSource, FpsCounter

_WIDTH = 64
_HEIGHT = 48
_FRAMES = 12


@pytest.fixture
def sample_video(tmp_path):
    # Setup
    """Write a small MJPG .avi and return its path as a string."""
    path = str(tmp_path / "sample.avi")
    writer = cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (_WIDTH, _HEIGHT),
    )
    for i in range(_FRAMES):
        frame = np.full((_HEIGHT, _WIDTH, 3), i * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    yield path

    # Teardown
    if os.path.exists(path):
        os.remove(path)

class TestFpsCounter:

    def test_stead_rate_is_measured(self):
        counter = FpsCounter(window_s=1.0)
        fps = 0.0
        for i in range(11):
            fps = counter.tick(now=i * 0.1)
        assert fps == pytest.approx(10.0, rel=0.01)

    def test_single_frame_reports_zero(self):
        assert FpsCounter().tick(now=5.0) == 0.0

    def test_old_samples_fall_out(self):
        counter = FpsCounter(window_s=1.0)
        counter.tick(now=0.0)
        counter.tick(now=0.1)
        assert counter.tick(now=10.0) == 0.0


class TestCameraSource:

    def test_open_read_all_frames_close(self, sample_video):
        source = CameraSource(sample_video)
        source.open()
        assert source.is_open
        assert source.width == _WIDTH
        assert source.height == _HEIGHT

        frames = 0
        while True:
            frame = source.read()
            if frame is None:
                break
            assert frame.shape == (_HEIGHT, _WIDTH, 3)
            frames += 1
        assert frames == _FRAMES

        source.close()
        assert not source.is_open

    def test_open_bad_path_raises(self):
        source = CameraSource("does_not_exist.avi")
        with pytest.raises(RuntimeError):
            source.open()

    def test_read_before_open_raises(self, sample_video):
        source = CameraSource(sample_video)
        with pytest.raises(RuntimeError):
            source.read()

    def test_close_is_repeat_safe(self, sample_video):
        source = CameraSource(sample_video)
        source.open()
        source.close()
        source.close()
