"""Prepare the FS2K face-photo / sketch dataset for Task 4.

The dataset is not downloaded by this script. Put the official FS2K folder at
data/FS2K so that it contains photo/, sketch/, anno_train.json and
anno_test.json (an extra folder level such as data/FS2K/FS2K/ is also fine).

What this script does:
1. Reads the official training and testing annotation files.
2. Splits the official training records into train / validation (15% val),
   stratified by sketch style, with a fixed seed. The official test records
   are left untouched and are never used for training or tuning.
3. Saves the split to manifests/fs2k_split.json.
4. Converts every photo and sketch to RGB, resizes both to 128x128 and saves
   each split as three NumPy files: <split>_photos.npy, <split>_sketches.npy
   and <split>_styles.npy. Row i of the three files is the same pair.
   Sketches are stored as 3-channel RGB, like the photos.
5. Prints a summary (counts per style, original sizes, how many sketches are
   greyscale) that is used to choose the generator's output channels.

Run from the repo root:
    python -m src.data.prepare_fs2k
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.config import PROJECT_ROOT, load_config


def find_root(fs2k_dir) -> Path:
    """Return the folder that contains anno_train.json.

    This is either fs2k_dir itself or a folder below it: unzipping the
    official file can add one or more extra levels (e.g. data/FS2K/FS2K/FS2K/).
    """
    fs2k_dir = Path(fs2k_dir)
    if (fs2k_dir / "anno_train.json").exists():
        return fs2k_dir

    # Look further down, preferring the shallowest match.
    if fs2k_dir.is_dir():
        matches = sorted(fs2k_dir.rglob("anno_train.json"), key=lambda p: (len(p.parts), str(p)))
        if matches:
            root = matches[0].parent
            print(f"Found the FS2K files in a sub-folder: {root}")
            return root

    raise FileNotFoundError(
        f"Could not find anno_train.json in {fs2k_dir} or in any folder "
        "inside it. Download FS2K (https://github.com/DengPingFan/FS2K) and "
        f"unpack it so that {fs2k_dir} contains photo/, sketch/, "
        "anno_train.json and anno_test.json."
    )


def read_annotations(json_path) -> list[dict]:
    """Read anno_train.json or anno_test.json.

    Returns one {"name": "photo1/image0110", "style": 0} dict per record, in
    file order. All other attributes (hair, gender, ...) are ignored.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return [{"name": r["image_name"], "style": int(r["style"])} for r in records]


def find_image(path_without_extension: Path) -> Path:
    """Return the file with a .jpg extension, or else .png (as the official
    FS2K tool does). Raise FileNotFoundError if neither exists.

    The extension may be written in capitals: FS2K contains files such as
    photo3/image0449.JPG. Windows ignores letter case but Linux (Kaggle,
    Docker) does not, so both spellings are tried explicitly.
    """
    for extension in [".jpg", ".png", ".JPG", ".PNG", ".jpeg", ".JPEG"]:
        path = path_without_extension.with_name(path_without_extension.name + extension)
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing image: {path_without_extension}.jpg (or .png, in any letter case)")


def resolve_pair(root, name) -> tuple[Path, Path]:
    """Return (photo_path, sketch_path) for a record name like "photo1/image0110".

    The official rule: replace "photo" with "sketch" and "image" with "sketch",
    so photo/photo1/image0110 -> sketch/sketch1/sketch0110.
    """
    root = Path(root)
    photo_relative = "photo/" + name
    sketch_relative = photo_relative.replace("photo", "sketch").replace("image", "sketch")
    photo_path = find_image(root / photo_relative)
    sketch_path = find_image(root / sketch_relative)
    return photo_path, sketch_path


def stratified_split(records, val_fraction: float = 0.15, seed: int = 42):
    """Split records into (train_records, val_records), stratified by style.

    For each style (in order 0, 1, 2) the records of that style are sorted by
    name, shuffled with one shared generator, and round(n_style * val_fraction)
    of them go to validation. Sorting first makes the result independent of
    the input order; the same seed always gives the same split.
    """
    rng = np.random.default_rng(seed)  # created once, used for every style
    train_records, val_records = [], []

    for style in sorted({r["style"] for r in records}):
        same_style = sorted((r for r in records if r["style"] == style), key=lambda r: r["name"])
        order = rng.permutation(len(same_style))
        num_val = round(len(same_style) * val_fraction)

        val_records += [same_style[i] for i in order[:num_val]]
        train_records += [same_style[i] for i in order[num_val:]]

    train_records = sorted(train_records, key=lambda r: r["name"])
    val_records = sorted(val_records, key=lambda r: r["name"])
    return train_records, val_records


def load_image(path, size: int = 128) -> np.ndarray:
    """Open an image, convert it to RGB and resize it to (size, size) with
    bicubic interpolation. Returns a uint8 array of shape (size, size, 3)."""
    with Image.open(path) as image:
        # Some images are greyscale or have an alpha channel; force 3 channels.
        image = image.convert("RGB")
        image = image.resize((size, size), Image.BICUBIC)
        return np.asarray(image, dtype=np.uint8)


def load_pair(photo_path, sketch_path, size: int = 128):
    """Load a photo and its sketch, both as uint8 arrays of shape (size, size, 3).

    Both go through exactly the same function (whole image, resized directly
    to size x size), so the face stays in the same place in both images.
    """
    return load_image(photo_path, size), load_image(sketch_path, size)


def image_info(path):
    """Return ((width, height), is_grey) for the original image file.
    is_grey is True when R == G == B for every pixel."""
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"))
        is_grey = bool(np.all(rgb[..., 0] == rgb[..., 1]) and np.all(rgb[..., 1] == rgb[..., 2]))
        return image.size, is_grey


def build_cache(root, records, out_dir, split, size: int = 128) -> dict:
    """Save <out_dir>/<split>_photos.npy, <split>_sketches.npy (uint8,
    (N, size, size, 3)) and <split>_styles.npy (int64, (N,)).

    Row i of every file belongs to records[i]. Returns a small summary of the
    original images: the set of photo sizes, the set of sketch sizes and the
    number of greyscale sketches.
    """
    out_dir = Path(out_dir)
    photos, sketches, styles = [], [], []
    summary = {"photo_sizes": set(), "sketch_sizes": set(), "grey_sketches": 0}

    for record in tqdm(records, desc=split):
        photo_path, sketch_path = resolve_pair(root, record["name"])
        photo, sketch = load_pair(photo_path, sketch_path, size)
        photos.append(photo)
        sketches.append(sketch)
        styles.append(record["style"])

        photo_size, _ = image_info(photo_path)
        sketch_size, sketch_is_grey = image_info(sketch_path)
        summary["photo_sizes"].add(photo_size)
        summary["sketch_sizes"].add(sketch_size)
        summary["grey_sketches"] += int(sketch_is_grey)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{split}_photos.npy", np.stack(photos).astype(np.uint8))
    np.save(out_dir / f"{split}_sketches.npy", np.stack(sketches).astype(np.uint8))
    np.save(out_dir / f"{split}_styles.npy", np.array(styles, dtype=np.int64))
    return summary


def main() -> None:
    """Make (or check) the split and build the caches."""
    parser = argparse.ArgumentParser(description="Prepare the FS2K dataset.")
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    # Paths in the config are relative to the repo root.
    cfg = load_config(PROJECT_ROOT / args.config)
    seed = cfg["seed"]
    image_size = cfg["image_size"]
    val_fraction = cfg["val_fraction"]["fs2k"]
    fs2k_dir = PROJECT_ROOT / cfg["data"]["fs2k_dir"]
    cache_dir = PROJECT_ROOT / cfg["data"].get("fs2k_cache_dir", "data/fs2k_128")
    manifests_dir = PROJECT_ROOT / cfg["data"]["manifests_dir"]

    root = find_root(fs2k_dir)

    # The official lists: anno_train is split into train/val, anno_test stays as it is.
    official_train = read_annotations(root / "anno_train.json")
    test_records = read_annotations(root / "anno_test.json")
    train_records, val_records = stratified_split(official_train, val_fraction, seed)

    split = {
        "seed": seed,
        "val_fraction": val_fraction,
        "image_size": image_size,
        "train": train_records,
        "val": val_records,
        "test": test_records,
    }

    split_file = manifests_dir / "fs2k_split.json"
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

    # Number of pairs per split and per style.
    for part in ["train", "val", "test"]:
        styles = [r["style"] for r in split[part]]
        per_style = {s: styles.count(s) for s in sorted(set(styles))}
        print(f"{part:5s}: {len(styles)} pairs, per style {per_style}")

    # One set of .npy files per split, rows in the same order as the JSON.
    photo_sizes, sketch_sizes, grey_sketches, total = set(), set(), 0, 0
    for part in ["train", "val", "test"]:
        summary = build_cache(root, split[part], cache_dir, part, image_size)
        photo_sizes |= summary["photo_sizes"]
        sketch_sizes |= summary["sketch_sizes"]
        grey_sketches += summary["grey_sketches"]
        total += len(split[part])
        print(f"Saved {part} caches in {cache_dir}")

    # Summary of the original files, used to choose the generator's out_channels.
    print(f"Original photo sizes (w, h):  {sorted(photo_sizes)}")
    print(f"Original sketch sizes (w, h): {sorted(sketch_sizes)}")
    print(f"Greyscale sketches (R == G == B everywhere): {grey_sketches} of {total}")


if __name__ == "__main__":
    main()
