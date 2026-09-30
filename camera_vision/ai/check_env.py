"""Environment check for AI dependencies.

Verifies PyTorch and GPU, laods DINOv3 from a local clone of
Meta's repo with pretrained weights, runs one forward pass, and
reports inference time.

Make sure the DINOv3 repository is cloned adjacent to this repository
and the "dinov3_vits16_pretrain_lvd1689m-08c60483.pth" weight file is
located in the models/ subdirectory within camera_vision/.
"""

import sys
import time
from importlib import import_module
from pathlib import Path


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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch {torch.__version__}")
    print(f"device: {device}", end="")
    if device.type == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print(" (no CUDA GPU found; analysis will run slower)")

    repo = Path(str(Path(__file__).resolve().parents[3] / "dinov3"))
    weights = Path(
        str(
            Path(__file__).resolve().parents[1]
            / "models"
            / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
        )
    )

    for label, path in (("dinov3 repo", repo), ("weights", weights)):
        if not path.exists():
            print(f"missing {label}: {path}")
            print("See the Setup section of the README.")
            return 1

    sys.path.insert(0, str(repo))
    backbones = import_module("dinov3.hub.backbones")
    model = getattr(backbones, "dinov3_vits16")(weights=str(weights))
    model = model.to(device).eval()
    print(f"loaded dinov3_vits16, {model.__class__.__name__}")

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


if __name__ == "__main__":
    sys.exit(main())