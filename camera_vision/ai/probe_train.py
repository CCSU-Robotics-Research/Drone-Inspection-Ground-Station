"""Train the defect probe on DINOv3 features with whatever data
is available in ``datasets/``.

* ``kaggle/`` - The surface-crack tiles.
* ``codebrim/original_dataset/`` - CODEBRIM bridge photos. Every
  annotated box is cropped out, and defect boxes keep their defect
  flags.

Each sample gets five 0/1 flags (Crack, Spallation,
Efflorescence, ExposedBars, CorrosionStain). A sample can have
several. One logistic regression per defect is fit on an
80/20 split and the probe is saved to ``models/probe_<backbone>.joblib``.
"""

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch

from ai import dino

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

DEFECTS = ("Crack", "Spallation", "Efflorescence", "ExposedBars",
           "CorrosionStain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DINOv3 defect probe."
    )
    parser.add_argument("--batch", type=int, default=64)
    return parser.parse_args()


def kaggle_items() -> list:
    """(path None, flags) for every Kaggle tile"""
    data = Path(__file__).resolve().parents[1] / "datasets/kaggle"
    items = []
    for name, crack in (("Negative", 0), ("Positive", 1)):
        for path in sorted((data / name).glob("*.jpg")):
            items.append((path, None, [crack, 0, 0, 0, 0]))
    return items


def codebrim_items() -> list:
    """(image path, box, flags) for every CODEBRIM box"""
    data = (
        Path(__file__).resolve().parents[1]
        / "datasets" / "codebrim" / "original_dataset"
    )
    images = {p.stem: p for p in (data / "images").iterdir()}
    items = []
    for xml_path in sorted((data / "annotations").glob("*.xml")):
        image = images.get(xml_path.stem)
        if image is None:
            continue
        for obj in ET.parse(xml_path).getroot().findall(".//object"):
            box = [int(float(c.text)) for c in obj.find(".//bndbox")]
            flags = [int(f.text) for f in obj.find(".//Defect")]
            items.append((image, box[:4], flags[1:6]))
    return items


def load(path: Path, box, cache: dict):
    """Image (or the box cut out of it); None if unreadable"""

    if cache.get("path") != path:
        cache["path"] = path
        cache["image"] = cv2.imread(
            str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
        )
    image = cache["image"]
    if image is None or box is None:
        return image
    x1, y1, x2, y2 = (max(0, v) for v in box)
    return image[y1:y2, x1:x2]


def extract_features(items, model, device, batch_size):
    feats, labels, batch = [], [], []
    cache = {}
    start = time.perf_counter()

    def flush():
        x = dino.preprocess(batch, device)
        feats.append(model(x).cpu().numpy())
        batch.clear()

    with torch.no_grad():
        for n, (path, box, flags) in enumerate(items, 1):
            crop = load(path, box, cache)
            if crop is None or crop.size == 0:
                continue
            batch.append(crop)
            labels.append(flags)
            if len(batch) == batch_size:
                flush()
            if n % 2000 == 0:
                rate = n / (time.perf_counter() - start)
                print(f"  {n}/{len(items)}, {rate:.0f} images/s")
        if batch:
            flush()
    return np.concatenate(feats), np.array(labels)


def main() -> int:
    args = parse_args()
    items = []

    kaggle_found = kaggle_items()
    codebrim_found = codebrim_items()
    print(f"kaggle: {len(kaggle_found)} tiles")
    print(f"codebrim: {len(codebrim_found)} items")
    items += kaggle_found
    items += codebrim_found

    device = dino.pick_device()
    print(f"device: {device}")
    model = dino.load_backbone()
    print(f"loaded {dino.BACKBONE}")

    feats, labels = extract_features(
        items, model, device, args.batch
    )
    print(f"features: {feats.shape}")
    for name, count in zip(DEFECTS, labels.sum(axis=0)):
        print(f"  {name}: {count} examples")

    x_tr, x_val, y_tr, y_val = train_test_split(
        feats, labels, test_size=0.2, random_state=0,
    )
    probe = OneVsRestClassifier(LogisticRegression(max_iter=2000))
    probe.fit(x_tr, y_tr)
    print(f"validation F1 per defect:")
    scores = f1_score(y_val, probe.predict(x_val), average=None,
                      zero_division=0)
    for name, score in zip(DEFECTS, scores):
        print(f"  {name}: {score:.4f}")

    out = dino.PROBE
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, out)
    print(f"saved probe: {out}")
    return 0


if __name__ == "__main__":
    main()
