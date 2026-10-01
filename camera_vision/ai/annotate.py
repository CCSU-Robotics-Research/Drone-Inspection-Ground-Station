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

_BOX_COLOR = (0, 165, 255)
_HEAT_COLOR = (0, 0, 255)


def draw(
    frame: np.ndarray,
    detections: Detections,
    *,
    heat_threshold: float = 0.5,
    heat_alpha: float = 0.45,
) -> np.ndarray:
    """Return a copy of ``frame`` with ``detections`` annotated.

    ``heat_threshold`` is the heatmap value above which a pixel
    is tinted. ``heat_alpha`` is the tint strength. The input
    frame is never modified.
    """
    result = frame.copy()  # Will be annotated and returned at end
    if detections.heatmap is not None:
        result = _draw_heatmap(
            result, detections.heatmap, heat_threshold, heat_alpha
        )
    for box, label, score in zip(
        detections.boxes, detections.labels, detections.scores
    ):
        _draw_box(result, box, f"{label} {score:.2f}")
    return result


def _draw_heatmap(
    result: np.ndarray,
    heatmap: np.ndarray,
    threshold: float,
    alpha: float,
) -> np.ndarray:
    heat = cv2.resize(
        heatmap.astype(np.float32),
        (result.shape[1], result.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    mask = heat >= threshold
    if not mask.any():
        return result
    tinted = result.copy()
    tinted[mask] = _HEAT_COLOR
    return cv2.addWeighted(result, 1.0 - alpha, tinted, alpha, 0)


def _draw_box(img: np.ndarray, box, text: str) -> None:
    x1, y1, x2, y2 = (int(v) for v in box)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 4)
    cv2.rectangle(img, (x1, y1), (x2, y2), _BOX_COLOR, 2)
    text_at = (x1, max(y1 - 8, 14))
    cv2.putText(img, text, text_at, cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (0, 0, 0), 4)
    cv2.putText(img, text, text_at, cv2.FONT_HERSHEY_SIMPLEX,
        0.6, _BOX_COLOR, 2)
