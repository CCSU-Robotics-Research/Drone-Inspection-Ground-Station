"""Train YOLO on CODEBRIM's 5 defect types."""

import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "codebrim" / "original_dataset"
DATA = ROOT / "datasets" / "codebrim_yolo"
MODELS = ROOT / "models"
DEFECTS = ("Crack", "Spallation", "Efflorescence", "ExposedBars",
           "CorrosionStain")


def yolo_labels(xml_path, width, height) -> list:
    """One 'class cx cy w h' line per defect per box"""
    lines = []
    for obj in ET.parse(xml_path).getroot().findall(".//object"):
        x1, y1, x2, y2 = (
            int(float(c.text)) for c in obj.find(".//bndbox")
        )
        flags = [int(f.text) for f in obj.find(".//Defect")][1:6]
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        for cls, flag in enumerate(flags):
            if flag:
                lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def prepare() -> None:
    images = sorted((SOURCE / "images").iterdir())
    random.Random(0).shuffle(images)
    n_val = len(images) * 15 // 100
    for split, files in (("val", images[:n_val]),
                         ("train", images[n_val:])):
        (DATA / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA / "labels" / split).mkdir(parents=True, exist_ok=True)
        for path in files:
            xml_path = SOURCE / "annotations" / (path.stem + ".xml")
            if not xml_path.exists():
                continue
            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION,
            )
            if image is None:
                continue
            height, width = image.shape[:2]
            lines = yolo_labels(xml_path, width, height)

            scale = 1024 / max(height, width)
            if scale < 1:
                image = cv2.resize(
                    image, (round(width * scale), round(height * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imwrite(
                str(DATA / "images" / split / (path.stem + ".jpg")), image
            )
            (DATA / "labels" / split / (path.stem + ".txt")).write_text(
                "\n".join(lines)
            )

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(DEFECTS))
    (DATA / "data.yaml").write_text(
        f"path: {DATA.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{names}\n"
    )


def main() -> None:
    if not (DATA / "data.yaml").exists():
        prepare()
    model = YOLO(str(MODELS / "yolo11n.pt"))
    model.train(data=str(DATA / "data.yaml"), epochs=100, imgsz=1024,
                project=str(MODELS / "yolo_runs"), name="codebrim")
    shutil.copy(model.trainer.best, MODELS / "yolo_codebrim.pt")
    print(f"saved {MODELS / 'yolo_codebrim.pt'}")


if __name__ == "__main__":
    main()
