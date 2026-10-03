"""Show live how the temperature tau changes the routing weights of Task 3.

Run from the repository root:

    python scripts/demo_temperature.py

It blurs a sample photo, runs the trained mixture-of-experts on it with
several temperatures, and prints the four routing weights each time.
A small tau makes the weights almost one-hot (nearly hard routing); a large
tau spreads them over the four branches. Nothing is saved or changed.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import src" works

from src.data import corruptions as C  # noqa: E402
from src.evaluation.eval_moe import load_moe  # noqa: E402
from src.models.soft_moe import BRANCH_NAMES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

model, checkpoint = load_moe(ROOT / "checkpoints" / "task3_moe.pt", torch.device("cpu"))
print(f"trained with tau = {checkpoint['tau']:.2f}\n")

photo = Image.open(ROOT / "backend" / "samples" / "pets" / "Birman_28.png").convert("RGB")
clean = torch.from_numpy(np.array(photo)).permute(2, 0, 1).float() / 255.0

cases = {
    "blur (kernel 5, sigma 1.5)": C.apply_corruption(clean, C.BLUR, {"kernel_size": 5, "sigma": 1.5}),
    "salt-and-pepper (p = 0.08)": C.apply_corruption(clean, C.SALT_PEPPER, {"prob": 0.08},
                                                     generator=torch.Generator().manual_seed(1)),
    "clean image": clean,
}

print(f"{'input':<30}{'tau':>5}   " + "  ".join(f"{name:>11}" for name in BRANCH_NAMES))
for label, image in cases.items():
    for tau in (0.3, 0.85, 3.0):
        model.set_tau(tau)
        with torch.no_grad():
            _, weights, _ = model(image.unsqueeze(0))
        print(f"{label:<30}{tau:>5.2f}   " + "  ".join(f"{w:>11.3f}" for w in weights[0].tolist()))
    print()
