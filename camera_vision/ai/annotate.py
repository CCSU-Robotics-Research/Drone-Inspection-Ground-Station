"""Drawing of detections onto frames.

``draw`` takes a frame and a ``Detections`` value and
return a new annotated frame. This leave the input untouched
and keeps this free of any model import, which is good
modular structure.

Colors are BGR (OpenCV order). Boxes and labels use amber over a
black outline pass. The heatmap overlays translucent red
wherever it exceeds the threshold.
"""

import cv2
import numpy as np

from ai.detect import Detections

# Per-defect box colors, unknown labels resort to default
_CLASS_COLORS = {
    "crack": (0, 165, 255),
    "spallation": (255, 140, 0),
    "efflorescence": (60, 190, 60),
    "exposedbars": (0, 0, 230),
    "corrosionstain": (200, 0, 200),
}


def draw(
    frame: np.ndarray,
    detections: Detections,
    *,
    heat_threshold: float = 0.5,
    heat_alpha: float = 0.45,
) -> np.ndarray:
    """Return a copy of ``frame`` with ``detections`` annotated.

    ``heat_threshold`` is the heatmap value above which a pixel
    is tinted. ``heat_alpha`` is the tint strength
    """
    result = frame.copy()

    # DINOv3 red heatmap
    if detections.heatmap is not None:
        heat = cv2.resize(
            detections.heatmap.astype(np.float32),
            (result.shape[1], result.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = heat >= heat_threshold
        if mask.any():
            tinted = result.copy()
            tinted[mask] = (0, 0, 255)  # red heatmap
            result = cv2.addWeighted(result, 1.0 - heat_alpha,
                                     tinted, heat_alpha, 0)

    # YOLO boxes
    fint = cv2.FONT_HERSHEY_SIMPLEX
    for box, label, score in zip(
        detections.boxes, detections.labels, detections.scores
    ):
        x1, y1, x2, y2 = (int(v) for v in box)
        color = _CLASS_COLORS.get(
            "".join(c for c in label.lower() if c.isalpha()),
            (0, 165, 255)  # default amber color
        )
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), 6)
        cv2.rectnagle(result, (x1, y1), (x2, y2), color, 3)

        text = f"{label} {score:.2f}"
        (tw, th), base = cv2.getTextSize(text, font, 1.0, 2)
        top = max(y1 - th - base - 6, 0)
        cv2.rectangle(result, (x1, top),
                      (x1 + tw + 8, top + th + base + 6), color, -1)
        cv2.putText(result, text, (x1 + 4, top + th + 3), font, 1.0,
                    (0, 0, 0), 2, cv2.LINE_AA)

    # Output annotated frame
    return result
