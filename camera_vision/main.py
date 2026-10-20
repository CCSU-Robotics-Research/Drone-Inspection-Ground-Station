"""Entry point for camera vision pipeline.

Opens a camera and shows the live feed in a window with
a measured-FPS overlay.

Usage::

    python main.py                    # open device 0 and streams
    python main.py --device 2         # another camera index
    python main.py --device clip.mp4  # play a recording instead
    python main.py --record           # records immediately
    python main.py -v                 # debug logging
    python main.py --no-stream        # blocks Unity streaming
    python main.py --annotate         # starts annotation immediately

Press r at anytime to toggle recording, and a to toggle live AI
annotation.

Press q or Esc in the video window to quit. You may have to change
the ``_CAPTURE_CARD`` field index below to get the correct source.
"""

import argparse
import logging
import signal
import sys
import time

import cv2

from ai.annotate import draw
from ai.detect import LiveAnalyzer, fuse
from commands import CommandListener
from read_feed import CameraSource, FpsCounter
from record import VideoRecorder
from stream import FrameStreamer

_CAPTURE_CARD = "2"

_LOG = logging.getLogger("main")


def _draw_rec(frame) -> None:
    """Indicator for recording when toggled at the bottom-right.
    Layered with a black in the background."""
    dot = (frame.shape[1] - 200, frame.shape[0] - 36)
    text_at = (frame.shape[1] - 175, frame.shape[0] - 20)
    cv2.circle(frame, dot, 17, (0, 0, 0), -1)
    cv2.circle(frame, dot, 17, (0, 0, 255), -1)
    cv2.putText(frame, "REC", text_at, cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (0, 0, 0), 8)
    cv2.putText(frame, "REC", text_at, cv2.FONT_HERSHEY_SIMPLEX,
                1.6, (0, 0, 255), 4)


def _draw_fps(frame, fps: float) -> None:
    """Measured-FPS overlay annotation over the camera feed."""
    cv2.putText(
        frame,
        f"{fps:.1f} FPS",
        (10, frame.shape[0] - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )


def _start_annotation():
    """Load the live models and starts background analyzer."""
    _LOG.info("Loading AI models for live annotation...")
    try:
        from ai.dino import DinoDefectDetector
        from ai.yolo import YoloDetector

        probe = DinoDefectDetector(backbone="dinov3_vits16")
        yolo = YoloDetector()
    except Exception:
        _LOG.exception("AI models failed to load; annotation off")
        return None

    def analyze(frame):
        return fuse(probe.analyze(frame), yolo.analyze(frame),
                    frame.shape[:2], 0.5)

    analyzer = LiveAnalyzer(analyze)
    analyzer.start()
    _LOG.info("Live annotation ready")
    return analyzer


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
        "--annotate",
        action="store_true",
        help="starts annotating immediately, toggle with a"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="debug-level logging",
    )
    return parser.parse_args()


def run(device, stream_enabled: bool, record_on_start: bool,
        annotate_on_start: bool) -> None:
    """Open the camera and display frames until quit/stall."""
    analyzer = _start_annotation()
    annotating = annotate_on_start and analyzer is not None

    source = CameraSource(device)
    source.open()
    recorder = VideoRecorder()

    # Video speed for file inputs
    frame_period = 1.0 / source.fps if (
        isinstance(device, str) and source.fps
    ) else 0.0

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

    # Toggles recording on/off
    def _toggle_recording() -> None:
        if recorder.is_recording:
            recorder.stop()
        else:
            recorder.start((source.width, source.height), source.fps)

    if record_on_start:
        _toggle_recording()

    # Turns live annotation on/off
    def _set_annotation(on: bool) -> None:
        nonlocal annotating
        if analyzer is None:
            _LOG.warning("Annotation unavailable since models did not load")
            return
        annotating = on
        if not on:
            analyzer.clear()
        _LOG.info("Annotation %s", "on" if on else "off")

    stop_requested = False

    # Handling shutdowns
    def _request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        _LOG.info("Shutdown requested")

    signal.signal(signal.SIGINT, _request_stop)

    commands = CommandListener({
        "record:toggle": _toggle_recording,
        "record:start": lambda: recorder.start(
            (source.width, source.height), source.fps
        ),
        "record:stop": recorder.stop,
        "annotate:toggle": lambda: _set_annotation(not annotating),
        "annotate:on": lambda: _set_annotation(True),
        "annotate:off": lambda: _set_annotation(False),
    })
    commands.start()

    fps_counter = FpsCounter()
    misses = 0
    next_frame = time.monotonic()

    try:
        while not stop_requested:
            if frame_period:
                next_frame += frame_period
                delay = next_frame - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_frame = time.monotonic()
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

            # Raw frame recording before annotation
            recorder.write(frame)

            # Send annotated window to stream and window
            if annotating:
                analyzer.submit(frame)
                frame = draw(frame, analyzer.latest,
                             heat_threshold=0.5)

            rec_visible = (
                recorder.is_recording
                and time.monotonic() % 1.0 < 0.5
            )

            # FPS indicator does not reach Unity but REC does
            if streamer is not None:
                stream_frame = frame.copy()
                stream_frame[stream_frame < 16] = 16
                if rec_visible:
                    _draw_rec(stream_frame)
                streamer.send(stream_frame)

            _draw_fps(frame, fps_counter.tick())

            if rec_visible:
                _draw_rec(frame)

            cv2.imshow("Camera Vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                _LOG.info("Quit requested")
                break
            elif key == ord("r"):
                _toggle_recording()
            elif key == ord("a"):
                _set_annotation(not annotating)
    finally:
        commands.stop()
        if analyzer is not None:
            analyzer.stop()
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
    run(device, args.stream, args.record, args.annotate)


if __name__ == "__main__":
    main()
