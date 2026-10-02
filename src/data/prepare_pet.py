"""Prepare the Oxford-IIIT Pet dataset for Tasks 1-3.

What this script does:
1. Downloads the dataset (skipped if it is already on disk).
2. Splits the official trainval list into 80% train / 20% validation
   with a fixed seed. The official test list is left untouched.
3. Saves the split to manifests/pet_split.json so every task uses the same one.
4. Converts every image to RGB, resizes it to 128x128 and saves the images
   as three NumPy files: train.npy, val.npy and test.npy.

Run from the repo root:
    python -m src.data.prepare_pet
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.config import PROJECT_ROOT, load_config


def download_pet(pet_dir) -> None:
    """Download the dataset so that it ends up in <pet_dir>/images and
    <pet_dir>/annotations. torchvision skips the download if the files exist."""
    # Imported here so the rest of this file can be used without torchvision.
    from torchvision.datasets import OxfordIIITPet

    # torchvision always creates a folder called "oxford-iiit-pet" inside
    # `root`, so we pass the parent folder of pet_dir as the root.
    root = Path(pet_dir).parent
    OxfordIIITPet(root=str(root), split="trainval", download=True)


def read_image_names(list_file) -> list[str]:
    """Read trainval.txt or test.txt and return the image names in file order.

    Each line looks like `Abyssinian_100 1 1 1`. We keep only the first
    column, which is the image name without the ".jpg" extension.
    """
    names = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue  # skip blank lines
            names.append(line.split()[0])
    return names


def make_split(names, val_fraction: float = 0.2, seed: int = 42):
    """Split the names into (train_names, val_names).

    The names are sorted first, so the result does not depend on the order
    they were given in. The same seed always gives the same split.
    """
    names = sorted(names)
    # A random ordering of the positions 0 .. len(names)-1.
    order = np.random.default_rng(seed).permutation(len(names))
    num_train = round(len(names) * (1 - val_fraction))

    train_names = [names[i] for i in order[:num_train]]
    val_names = [names[i] for i in order[num_train:]]
    return sorted(train_names), sorted(val_names)


def load_image(path, size: int = 128) -> np.ndarray:
    """Open an image, convert it to RGB and resize it to (size, size).

    The image is resized directly (no cropping), so non-square images are
    squashed. Returns a uint8 array of shape (size, size, 3).
    """
    with Image.open(path) as image:
        # Some images are greyscale or have an alpha channel; force 3 channels.
        image = image.convert("RGB")
        image = image.resize((size, size), Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def build_cache(names, images_dir, out_file, size: int = 128):
    """Load <images_dir>/<name>.jpg for every name and save them all in one
    .npy file of shape (N, size, size, 3). Row i belongs to names[i].
    Returns the shape of the saved array."""
    images_dir = Path(images_dir)
    out_file = Path(out_file)

    images = []
    for name in tqdm(names, desc=out_file.name):
        images.append(load_image(images_dir / f"{name}.jpg", size))
    array = np.stack(images).astype(np.uint8)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_file, array)
    return array.shape


def main() -> None:
    """Download the dataset, make (or check) the split and build the caches."""
    parser = argparse.ArgumentParser(description="Prepare the Oxford-IIIT Pet dataset.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    # Paths in the config are relative to the repo root, so joining them with
    # PROJECT_ROOT makes the script work on both Windows and Kaggle (Linux).
    cfg = load_config(PROJECT_ROOT / args.config)
    seed = cfg["seed"]
    image_size = cfg["image_size"]
    val_fraction = cfg["val_fraction"]["pet"]
    pet_dir = PROJECT_ROOT / cfg["data"]["pet_dir"]
    cache_dir = PROJECT_ROOT / cfg["data"].get("pet_cache_dir", "data/pet_128")
    manifests_dir = PROJECT_ROOT / cfg["data"]["manifests_dir"]

    if not args.skip_download:
        download_pet(pet_dir)

    # The official lists: trainval is our development data, test stays as it is.
    trainval_names = read_image_names(pet_dir / "annotations" / "trainval.txt")
    test_names = read_image_names(pet_dir / "annotations" / "test.txt")
    train_names, val_names = make_split(trainval_names, val_fraction, seed)

    split = {
        "seed": seed,
        "val_fraction": val_fraction,
        "image_size": image_size,
        "train": train_names,
        "val": val_names,
        "test": test_names,
    }

    split_file = manifests_dir / "pet_split.json"
    if split_file.exists():
        # The split is fixed. Never overwrite it: only check that it still
        # matches what we would produce now, then reuse it.
        with open(split_file, "r", encoding="utf-8") as f:
            saved_split = json.load(f)
        if saved_split != split:
            raise ValueError(
                f"{split_file} does not match the split produced now. "
                "The saved split must not change; check the seed, "
                "val_fraction, image_size and the dataset files."
            )
        split = saved_split
        print(f"Reusing existing split: {split_file}")
    else:
        manifests_dir.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=1)
        print(f"Saved split: {split_file}")

    print(f"train: {len(split['train'])} images")
    print(f"val:   {len(split['val'])} images")
    print(f"test:  {len(split['test'])} images")

    # One .npy file per part, rows in the same order as the names in the JSON.
    images_dir = pet_dir / "images"
    for part in ["train", "val", "test"]:
        out_file = cache_dir / f"{part}.npy"
        shape = build_cache(split[part], images_dir, out_file, image_size)
        print(f"{out_file}: shape {shape}")


if __name__ == "__main__":
    main()
