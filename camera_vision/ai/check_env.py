"""Environment check for AI dependencies.

Verifies PyTorch and GPU, laods DINOv3 from a local clone of
Meta's repo with pretrained weights, runs one forward pass, and
reports inference time.

Make sure the DINOv3 repository is cloned adjacent to this repository
and the "dinov3_vits16_pretrain_lvd1689m-08c60483.pth" weight file is
located in the models/ subdirectory within camera_vision/.

Make sure the DINOv3 probe and YOLO models are trained and ready.
"""

import numpy as np
import sys
import time


def main() -> int:

    try:
        import torch
    except ImportError:
        print(
            "torch is not installed. Install it per "
            "https://pytorch.org/get-started/locally, then "
            'run: pip install -e ".[dev,ai]"'
        )
        return 1
    from ai import dino

    # DINOv3 backbone verification

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}")
    print(f"device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (no CUDA GPU found; analysis will run slower)")

    model = dino.load_backbone()
    print(f"loaded {dino.BACKBONE}, {model.__class__.__name__}")

    x = torch.zeros(1, 3, 224, 224, device=device)
    with torch.no_grad():
        out = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

    print(f"forward pass OK, output shape {tuple(out.shape)}")
    print(f"{elapsed / 10 * 1000:.1f} ms per 224x224 pass, 10 passes")

    models = Path(__file__).resolve().parents[1] / "models"
    for name in (dino.PROBE.name, yolo.MODEL.name):
        found = (models / name).exists()

    # DINOv3 probe and YOLO model verification

    try:
        from ultralytics import YOLO
        from ai import yolo
    except ImportError:
        print('ultralytics is not installed, run pip install -e ".[dev,ai]"')
        return 1

    for path in (dino.PROBE, yolo.MODEL):
        found = path.exists()
        print(f"{path.name}: {'found' if found else 'not trained yet'}")

    if yolo.MODEL.exist():
        yolo_model = YOLO(str(yolo.MODEL))
        frame = np.zeros((720, 1280, 3), np.uint8)
        yolo_model(frame, verbose=False)
        start = time.perf_counter()
        for _ in range(10):
            yolo_model(frame, verbose=False)
        elapsed = time.perf_counter() - start
        print(f"YOLO OK, {elapsed / 10 * 1000:.1f} ms per 720p frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
