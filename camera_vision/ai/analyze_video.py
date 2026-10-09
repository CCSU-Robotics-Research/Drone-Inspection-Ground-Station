"""Annotate a recorded video with the crack probe.

Runs ``DinoCrackDetector`` on every Nth frame of a video,
draws the most recent detections on every frame, and writes
an annotated copy plus a per-analysis JSON summary.

To avoid calculating manual directory paths,
Usage::

    python -m ai.analyze_video recordings\\video.avi
    python -m ai.analyze_video video.mp4 --frame-interval 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from ai import annotate, dino
from ai.detect import Detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate a video with crack detections."
    )
    parser.add_argument("video", help="input video file")
    parser.add_argument(
        "--frame-interval", type=int, default=5,
        help="Analyze every Nth frame. Default 5",
    )
    parser.add_argument(
        "--fine", action="store_true",
        help="enable fine 16 px heatmap"
    )
    parser.add_argument("--heat-threshold", type=float,
                        default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    src = Path(args.video)
    if not src.exists():
        print(f"missing video: {src}")
        return 1
    out_path = src.with_name(src.stem + "_annotated.avi")
    json_path = out_path.with_suffix(".json")

    detector = dino.DinoCrackDetector(fine=args.fine)
    print(f"device: {detector.device}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        print(f"could not open video: {src}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"MJPG"),
        fps, (width, height),
    )
    writer.set(cv2.VIDEOWRITER_PROP_QUALITY, 95)

    detections = Detections()
    records = []
    index = 0
    start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % args.frame_interval == 0:
            detections = detector.analyze(frame)
            heat = detections.heatmap
            records.append({
                "frame": index,
                "heat_max": round(float(heat.max()), 4),
                "heat_mean": round(float(heat.mean()), 4),
                "tiles_above_threshold": int(
                    (heat >= args.heat_threshold).sum()
                ),
            })
        writer.write(annotate.draw(
            frame, detections,
            heat_threshold=args.heat_threshold,
        ))
        index += 1
        if index % 100 == 0:
            rate = index / (time.perf_counter() - start)
            print(f"  {index} frames, {rate:.1f} fps")

    cap.release()
    writer.release()
    json_path.write_text(json.dumps(records, indent=2))
    print(f"wrote {out_path}")
    print(f"wrote {json_path}, {len(records)} analyses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
