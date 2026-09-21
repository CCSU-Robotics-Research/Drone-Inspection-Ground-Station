"""Entry point for camera vision pipeline.

Opens a camera and shows the live feed in a window with
a measured-FPS overlay.

Usage::

    python main.py                    # open device 0
    python main.py --device 2         # another camera index
    python main.py --device clip.mp4  # play a recording instead
    python main.py --record           # records immediately
    python main.py -v                 # debug logging
    python main.py --stream           # sends frames to Unity

Press r at anytime to toggle recording.

Press q or Esc in the video window to quit. You may have to change
the ``_CAPTURE_CARD`` field index below to get the correct source.
"""

import argparse
import logging
import sys

import cv2

from read_feed import CameraSource, FpsCounter
from record import VideoRecorder
from stream import FrameStreamer

_CAPTURE_CARD = "2"

_LOG = logging.getLogger("main")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Ground station camera feed reading and display."
    )
    parser.add_argument(
        "--device",
        default=_CAPTURE_CARD,
        help="camera index like 2, or a video file path"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="starts recording immediately, toggle with r"
    )
    parser.add_argument(
        "--no-stream",
        dest="stream",
        action="store_false",
        help="does not send frames over TCP for Unity",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug-level logging",
    )
    return parser.parse_args()


def run(device, stream_enabled: bool, record_on_start: bool) -> None:
    """Open the camera and display frames until quit/stall."""
    source = CameraSource(device)
    source.open()
    recorder = VideoRecorder()

    streamer = None
    if stream_enabled:
        streamer = FrameStreamer()
        try:
            streamer.start()
        except OSError:
            source.close()
            sys.exit(
                "[CAMERA] port 5010 already in use, is another "
                "camera_vision instance running? Use --no-stream "
                "for a second local viewer"
            )

    def _toggle_recording() -> None:
        if recorder.is_recording:
            recorder.stop()
        else:
            recorder.start((source.width, source.height), source.fps)

    if record_on_start:
        _toggle_recording()

    fps_counter = FpsCounter()
    misses = 0

    try:
        while True:
            frame = source.read()
            if frame is None:
                misses += 1
                if misses >= 30:
                    _LOG.warning(
                        "Stream ended or camera stalled; exiting"
                    )
                    break
                continue
            misses = 0

            # Raw frame recording
            recorder.write(frame)

            if streamer is not None:
                # Copy so FPS overlay (code below in OpenCV) doesn't
                # appear in HoloLens
                streamer.send(frame.copy())

            fps = fps_counter.tick()
            cv2.putText(
                frame,
                f"{fps:.1f} FPS",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            if recorder.is_recording:
                cv2.circle(
                    frame,
                    (frame.shape[1] - 105, 26),
                    8,
                    (0, 0, 255),
                    -1,
                )
                cv2.putText(
                    frame,
                    "REC",
                    (frame.shape[1] - 90, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Camera Vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                _LOG.info("Quit requested")
                break
            elif key == ord("r"):
                _toggle_recording()
    except KeyboardInterrupt:
        _LOG.info("Interrupted")
    finally:
        recorder.stop()
        if streamer is not None:
            streamer.stop()
        source.close()
        cv2.destroyAllWindows()
        _LOG.info("Camera closed")


def main() -> None:
    """Display the camera feed."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = int(args.device) if args.device.isdigit() else args.device
    run(device, args.stream, args.record)


if __name__ == "__main__":
    main()
