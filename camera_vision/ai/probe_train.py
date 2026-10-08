"""Train the crack probe on the Kaggle surface-crack tiles.

Extracts DINOv3 features for every tile in the dataset's
``Positive`` and ``Negative`` folders, fits a logistic-regression
probe on an 80/20 stratified split, prints the validation
accuracy, and saves the probe to ``models/probe.joblib``.
"""

import argparse
import sys
import time
import torch
from pathlib import Path

import cv2
import joblib
import numpy as np

from ai import dino

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


_CLASSES = (("Negative", 0), ("Positive", 1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the DINOv3 crack probe."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="use at most N images per class for quick runs",
    )
    parser.add_argument("--batch", type=int, default=64)
    return parser.parse_args()


def gather(root: Path, limit) -> list:
    items = []
    for name, label in _CLASSES:
        folder = root / name
        if not folder.is_dir():
            print(f"missing class folder: {folder}")
            sys.exit(1)
        files = sorted(
            p for p in folder.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if limit:
            files = files[:limit]
        items += [(p, label) for p in files]
        print(f"{name}: {len(files)} images")
    return items


def extract_features(items, model, device, batch_size):
    feats, labels = [], []
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            images = [cv2.imread(str(p)) for p, _ in chunk]
            x = dino.preprocess(images, device)
            feats.append(model(x).cpu().numpy())
            labels += [label for _, label in chunk]
            done = i + len(chunk)
            if done % (batch_size * 20) == 0:
                rate = done / (time.perf_counter() - start)
                print(f"  {done}/{len(items)} "
                      f"({rate:.0f} images/s)")
    return np.concatenate(feats), np.array(labels)


def main() -> int:
    args = parse_args()

    device = dino.pick_device()
    print(f"device: {device}")
    model = dino.load_backbone()
    print("loaded dinov3_vits16")

    items = gather(Path(__file__).resolve().parents[1] / "datasets/kaggle",
                   args.limit)
    feats, labels = extract_features(
        items, model, device, args.batch
    )
    print(f"features: {feats.shape}")

    x_tr, x_val, y_tr, y_val = train_test_split(
        feats, labels, test_size=0.2, random_state=0,
        stratify=labels,
    )
    probe = LogisticRegression(max_iter=2000)
    probe.fit(x_tr, y_tr)

    acc_tr = probe.score(x_tr, y_tr)
    acc_val = probe.score(x_val, y_val)
    print(f"train accuracy: {acc_tr:.4f}")
    print(f"validation accuracy: {acc_val:.4f} "
          f"({len(y_val)} tiles)")

    out = dino.PROBE
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(probe, out)
    print(f"saved probe: {out}")
    return 0


if __name__ == "__main__":
    main()
