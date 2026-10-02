"""Save a grid showing every test corruption at every severity.

Each row is one test image; the columns are the ten manifest entries of that
image (clean, then salt-and-pepper, blur and occlusion at low/medium/high).

Run:  python -m src.evaluation.preview_corruptions
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt

from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, load_images
from src.utils.config import PROJECT_ROOT, load_config

ENTRIES_PER_IMAGE = 10


def main():
    parser = argparse.ArgumentParser(description="Preview the fixed test corruptions.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--out", default="report/figures/corruption_preview.png")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    images = load_images(PROJECT_ROOT / cfg["data"]["pet_cache_dir"], "test")
    entries = load_manifest(PROJECT_ROOT / cfg["data"]["manifests_dir"] / "pet_test_manifest.jsonl")
    dataset = PetManifestDataset(images, entries)

    rows, cols = args.num_images, ENTRIES_PER_IMAGE
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.75))
    for row in range(rows):
        for col in range(cols):
            index = row * ENTRIES_PER_IMAGE + col
            corrupted, _, _ = dataset[index]
            entry = entries[index]
            ax = axes[row, col]
            ax.imshow(corrupted.permute(1, 2, 0).numpy())
            ax.axis("off")
            if row == 0:
                title = "clean" if entry["severity"] == "none" else f"{entry['condition']}\n{entry['severity']}"
                ax.set_title(title, fontsize=8)

    fig.tight_layout()
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
