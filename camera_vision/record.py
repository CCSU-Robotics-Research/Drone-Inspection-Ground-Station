"""Video recording for offline AI analysis.

Records raw camera feed to disk. At MJPG quality 95 with
OpenCV, estimated sizestorage is 0.3-0.5GB per minute for
720p30 video.

Lossless recording can be enabled by switching ``_FOURCC`` to
``"FFV1"``.

The recorder is toggled from the window hotkey, making start,
stop, and write serialized by a lock.
"""

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2

_LOG = logging.getLogger(__name__)

_FOURCC = "MJPG"
_EXTENSION = ".avi"
_DEFAULT_DIR = "recordings"


class VideoRecorder:
    """Writes contiguous frames to an auto-named video file."""

    def __init__(self, output_dir: str = _DEFAULT_DIR) -> None:
        self._dir = Path(output_dir)
        self._lock = threading.Lock()

        self._writer = None
        self._path: Optional[Path] = None
        self._frames = 0
        self._started_at = 0

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._writer is not None

    @property
    def path(self) -> Optional[Path]:
        """The latest recording path."""
        with self._lock:
            return self._path

    def start(
        self, frame_size: Tuple[int, int], fps: float
    ) -> Optional[Path]:
        """Open a new recording and return its path.

        ``frame_size`` is (width, height). A repeated start while
        already recording is ignored.
        """
        with self._lock:
            if self._writer is not None:
                _LOG.info("Already recording to %s", self._path)
                return self._path

            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._unique_path()

            if fps <= 0:
                fps = 30.0
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*_FOURCC),
                fps,
                frame_size,
            )
            if not writer.isOpened():
                raise RuntimeError(
                    f"could not start recording at {path}"
                )
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)

            self._writer = writer
            self._path = path
            self._frames = 0
            self._started_at = time.monotonic()

            _LOG.info(
                "Recording to %s (%dx%d @ %.1f fps, %s)",
                path, frame_size[0], frame_size[1], fps, _FOURCC,
            )
            return path

    def write(self, frame) -> None:
        """Write one frame. If not recording, this no-ops."""
        with self._lock:
            if self._writer is None:
                return
            self._writer.write(frame)
            self._frames += 1

    def stop(self) -> None:
        """Close the recording."""
        with self._lock:
            if self._writer is None:
                return
            duration = time.monotonic() - self._started_at
            self._writer.release()
            self._writer = None
            _LOG.info(
                "Recording stopped: %s (%d frames, %.1f s)",
                self._path, self._frames, duration,
            )

    def _unique_path(self) -> Path:
        """Timestamp-named path to prevent overwrites."""
        stamp = datetime.now().strftime("rec_%Y%m%d_%H%M%S")
        path = self._dir / f"{stamp}{_EXTENSION}"
        counter = 2
        while path.exists():
            path = self._dir / f"{stamp}_{counter}{_EXTENSION}"
            counter += 1
        return path
