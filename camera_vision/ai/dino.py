"""DINOv3 backbone and the crack-probe detector.

Loads the DINOv3 backbone module from a local clone, plus
the LVD-1689M normalization baked into preprocessing.

``DinoCrackDetector`` implements the ``Detector`` protocol by
sliding a 224 px window over the frame, scoring each tile with
the trained probe, and returns the scores as a coarse heatmap that
``ai.annotate`` upscales.

The backbone exposes 2 feature levels. ``model(x)`` returns one
summary vector per tile. ``patch_tokens`` returns one vector per
16 px patch (aka 196 per tile), which will be important for YOLO
to add bounding boxes.
"""

from importlib import import_module
from pathlib import Path
import sys

import cv2
import joblib
import numpy as np
import sys
import torch

from ai.detect import Detections

# Constants for repeated use or in other files
PROBE = (
    Path(__file__).resolve().parents[1] / "models" / "probe.joblib"
)

TILE = 224
STRIDE = 112
PATCH = 16
PATCHES = TILE // PATCH


def pick_device() -> "torch.device":
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_backbone():
    """Load DINOv3 ViT-S/16 backbone in eval mode."""
    repo = Path(__file__).resolve().parents[3] / "dinov3"
    weights = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )
    device = pick_device()
    sys.path.insert(0, str(repo))
    backbones = import_module("dinov3.hub.backbones")
    model = getattr(backbones, "dinov3_vits16")(weights=str(weights))
    return model.to(device).eval()


def preprocess(images_bgr, device) -> "torch.Tensor":
    """Takes BGR uint8 arrays and normalizes a float batch."""
    batch = []
    for img in images_bgr:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (TILE, TILE):
            rgb = cv2.resize(rgb, (TILE, TILE),
                             interpolation=cv2.INTER_AREA)
        batch.append(rgb)
    x = torch.from_numpy(np.stack(batch)).to(device)
    x = x.permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor((0.485, 0.456, 0.406),
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225),
                       device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def patch_tokens(model, x) -> "torch.Tensor":
    """Per patch features (batch, 196, 384) for 224 px input.
    Will be used for detection head outputs later.
    """
    return model.forward_features(x)["x_norm_patchtokens"]


def assemble(
    blocks: np.ndarray,
    rows: int,
    cols: int
) -> np.ndarray:
    """Stitch (rows*cols, P, P) blocks into one (rows*P, cols*P)
    grid, row-major order, matching tile order from analyze().
    Will be used for detection head outputs later.
    """
    p = blocks.shape[1]
    grid = blocks.reshape(rows, cols, p, p)
    grid = grid.transpose(0, 2, 1, 3)
    return grid.reshape(rows * p, cols * p)


def tile_starts(
    length: int,
    tile: int = TILE,
    stride: int = STRIDE,
) -> list:
    """Window starts offsets covering [0, length)."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


class DinoCrackDetector:
    """DINOv3 features + trained crack probe."""

    def __init__(self, batch_size: int = 64):
        self.device = pick_device()
        self.model = load_backbone()
        self.fine = fine
        self.probe = joblib.load(PROBE_FINE if fine else PROBE)
        self.batch_size = batch_size

    def analyze(self, frame: np.ndarray) -> Detections:
        h, w = frame.shape[:2]
        stride = TILE if self.fine else STRIDE
        ys = tile_starts(h, TILE, stride)
        xs = tile_starts(w, TILE, stride)
        tiles = [frame[y:y + TILE, x:x + TILE]
                 for y in ys for x in xs]

        scores = []
        with torch.no_grad():
            for i in range(0, len(tiles), self.batch_size):
                x = preprocess(tiles[i:i + self.batch_size],
                               self.device)
                feats = self.model(x).cpu().numpy()
                scores.append(self.probe.predict_proba(feats)[:, 1])

        heat = np.concatenate(scores).astype(np.float32)
        heat = heat.reshape(len(ys), len(xs))
        return Detections(heatmap=heat)
