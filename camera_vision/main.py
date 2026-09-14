"""Entry point for camera vision pipeline.

Opens a camera and shows the live feed in a window with
a measured-FPS overlay.

Usage::

    python main.py                    # open device 0
    python main.py --device 2         # another camera index
    python main.py --device clip.mp4  # play a recording instead
    python main.py --probe            # list camera indices that open
    python main.py -v                 # debug logging

Press q or Esc in the video window to quit. You may have to change
the ``_CAPTURE_CARD`` field index below to get the correct source.
"""

import argparse
import logging

import cv2

from read_feed import BACKEND, CameraSource, FpsCounter

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
        "-v",
        "--verbose",
        action="store_true",
        help="debug-level logging",
    )
    return parser.parse_args()


def run(device) -> None:
    """Open the camera and display frames until quit/stall."""
    source = CameraSource(device)
    source.open()

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

            cv2.imshow("Camera Vision", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                _LOG.info("Quit requested")
                break
    except KeyboardInterrupt:
        _LOG.info("Interrupted")
    finally:
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
    run(device)


if __name__ == "__main__":
    main()
