"""Reads a camera feed from a source.

``CameraSource`` reads from the specified capture device
and wraps ``cv2.VideeoCapture``. ``device`` is either an
index to the specified webcam or capture card, or a string
path to a video file.
"""

import logging
import time
from collections import deque
from typing import Optional, Union

import cv2

_LOG = logging.getLogger(__name__)
BACKEND = cv2.CAP_ANY


class FpsCounter:
    """Measures achieved FPS over a sliding window."""

    def __init__(self, window_s: float = 1.0) -> None:
        self._window_s = window_s
        self._stamps: deque = deque()

    def tick(self, now: Optional[float] = None) -> float:
        """Record one frame and return the FPS estimate."""
        if now is None:
            now = time.monotonic()
        self._stamps.append(now)

        cutoff = now - self._window_s
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.popleft()

        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        if span <= 0:
            return 0.0
        return (len(self._stamps) - 1) / span


class CameraSource:
    """Reads frames from specified capture device."""

    def __init__(self, device: Union[int, str]) -> None:
        self._device = device
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def width(self) -> int:
        return self._cap and int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0.0

    @property
    def height(self) -> int:
        return self._cap and int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0.0

    @property
    def fps(self) -> float:
        return self._cap and float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0

    def open(self) -> None:
        """Opens the device."""
        self._cap = cv2.VideoCapture(self._device, BACKEND)

        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"could not open camera {self._device!r}. Check the "
                f"connection, try another index with --device."
            )

        _LOG.info(
            "Opened camera %r: %dx%d @ %.1f fps",
            self._device,
            self.width,
            self.height,
            self.fps
        )

    def read(self):
        """Return the next frame if available. ``None`` indicates EOF
        for video source, or a stalled/disconnected camera.
        """
        if self._cap is None:
            raise RuntimeError("CameraSource.read() before open()")

        ok, frame = self._cap.read()
        return None if not ok else frame

    def close(self) -> None:
        """Release the device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
