"""Tests for src/data/prepare_fs2k.py and src/data/fs2k_dataset.py.
A tiny fake FS2K folder is built in tmp_path; no real dataset needed."""

import json
import random
import sys

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import prepare_fs2k
from src.data.fs2k_dataset import FS2KDataset, to_unit_range
from src.data.prepare_fs2k import (
    build_cache,
    find_root,
    load_pair,
    read_annotations,
    resolve_pair,
    stratified_split,
)


def colour_of(index):
    """A distinct plain colour for image number `index`."""
    return ((index * 37) % 256, (index * 91) % 256, (index * 53) % 256)


def make_fake_fs2k(root):
    """Write a fake FS2K tree: 12 train + 6 test records over the three styles.

    Mixes .jpg and .png, includes a greyscale photo and an RGBA sketch, and
    uses different original sizes for photos and sketches. Returns the
    (train, test) annotation lists as written to the JSON files.
    """
    train, test = [], []
    for i in range(18):
        folder = i % 3 + 1  # photo1, photo2, photo3
        name = f"photo{folder}/image{i:04d}"
        record = {"image_name": name, "style": i % 3, "hair": 0, "gender": 1}
        (train if i < 12 else test).append(record)

        photo_dir = root / "photo" / f"photo{folder}"
        sketch_dir = root / "sketch" / f"sketch{folder}"
        photo_dir.mkdir(parents=True, exist_ok=True)
        sketch_dir.mkdir(parents=True, exist_ok=True)

        if i == 0:
            photo = Image.new("L", (250, 250), 120)  # greyscale photo
        else:
            photo = Image.new("RGB", (250, 300), colour_of(i))
        photo.save(photo_dir / f"image{i:04d}{'.png' if i % 4 == 0 else '.jpg'}")

        if i == 1:
            sketch = Image.new("RGBA", (200, 250), (90, 90, 90, 255))  # RGBA sketch
            sketch.save(sketch_dir / f"sketch{i:04d}.png")
        else:
            sketch = Image.new("L", (200, 250), (i * 13) % 256)
            sketch.save(sketch_dir / f"sketch{i:04d}{'.png' if i % 5 == 0 else '.jpg'}")

    with open(root / "anno_train.json", "w", encoding="utf-8") as f:
        json.dump(train, f)
    with open(root / "anno_test.json", "w", encoding="utf-8") as f:
        json.dump(test, f)
    return train, test


def fake_records(counts):
    """Records like the real ones, `counts[s]` of them for style s."""
    records = []
    for style, n in enumerate(counts):
        records += [{"name": f"photo{style + 1}/image{style}{i:04d}", "style": style} for i in range(n)]
    return records


# ---------- find_root / read_annotations / resolve_pair ----------

def test_find_root_flat_and_nested(tmp_path):
    flat = tmp_path / "flat"
    make_fake_fs2k(flat)
    assert find_root(flat) == flat

    nested = tmp_path / "nested"
    make_fake_fs2k(nested / "FS2K")
    assert find_root(nested) == nested / "FS2K"


def test_find_root_missing_gives_helpful_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="anno_train.json"):
        find_root(tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="FS2K"):
        find_root(tmp_path / "does_not_exist")


def test_read_annotations_keeps_name_style_and_order(tmp_path):
    train, _ = make_fake_fs2k(tmp_path)
    records = read_annotations(tmp_path / "anno_train.json")
    assert records == [{"name": r["image_name"], "style": r["style"]} for r in train]


def test_resolve_pair_maps_photo_to_sketch(tmp_path):
    make_fake_fs2k(tmp_path)
    # image0000: photo is .png (0 % 4 == 0), sketch is .png (0 % 5 == 0)
    photo, sketch = resolve_pair(tmp_path, "photo1/image0000")
    assert photo == tmp_path / "photo" / "photo1" / "image0000.png"
    assert sketch == tmp_path / "sketch" / "sketch1" / "sketch0000.png"
    # image0003: both .jpg, folder 1
    photo, sketch = resolve_pair(tmp_path, "photo1/image0003")
    assert photo == tmp_path / "photo" / "photo1" / "image0003.jpg"
    assert sketch == tmp_path / "sketch" / "sketch1" / "sketch0003.jpg"
    # image0001: photo .jpg, sketch .png, folder 2
    photo, sketch = resolve_pair(tmp_path, "photo2/image0001")
    assert photo == tmp_path / "photo" / "photo2" / "image0001.jpg"
    assert sketch == tmp_path / "sketch" / "sketch2" / "sketch0001.png"


def test_resolve_pair_prefers_jpg(tmp_path):
    make_fake_fs2k(tmp_path)
    Image.new("RGB", (10, 10)).save(tmp_path / "sketch" / "sketch1" / "sketch0003.png")
    _, sketch = resolve_pair(tmp_path, "photo1/image0003")
    assert sketch.suffix == ".jpg"


def test_resolve_pair_missing_sketch_raises(tmp_path):
    make_fake_fs2k(tmp_path)
    (tmp_path / "sketch" / "sketch1" / "sketch0003.jpg").unlink()
    with pytest.raises(FileNotFoundError, match="sketch0003"):
        resolve_pair(tmp_path, "photo1/image0003")


# ---------- stratified_split ----------

def test_split_counts_per_style():
    counts = [100, 50, 31]
    train, val = stratified_split(fake_records(counts), 0.15, 42)
    for style, n in enumerate(counts):
        num_val = sum(r["style"] == style for r in val)
        num_train = sum(r["style"] == style for r in train)
        assert num_val == round(n * 0.15)
        assert num_train == n - round(n * 0.15)


def test_split_disjoint_complete_and_sorted():
    records = fake_records([40, 30, 20])
    train, val = stratified_split(records)
    train_names = [r["name"] for r in train]
    val_names = [r["name"] for r in val]
    assert set(train_names) & set(val_names) == set()
    assert sorted(train_names + val_names) == sorted(r["name"] for r in records)
    assert train_names == sorted(train_names)
    assert val_names == sorted(val_names)


def test_split_repeatable_and_ignores_input_order():
    records = fake_records([40, 30, 20])
    shuffled = list(records)
    random.Random(0).shuffle(shuffled)
    assert shuffled != records
    assert stratified_split(records) == stratified_split(records)
    assert stratified_split(shuffled) == stratified_split(records)
    assert stratified_split(records, seed=42) != stratified_split(records, seed=7)


def test_split_every_style_in_validation():
    _, val = stratified_split(fake_records([4, 4, 4]))
    assert sorted({r["style"] for r in val}) == [0, 1, 2]


# ---------- load_pair / build_cache ----------

def test_load_pair_greyscale_and_rgba(tmp_path):
    make_fake_fs2k(tmp_path)
    for name in ["photo1/image0000", "photo2/image0001"]:  # grey photo, RGBA sketch
        photo, sketch = load_pair(*resolve_pair(tmp_path, name))
        for array in [photo, sketch]:
            assert array.shape == (128, 128, 3)
            assert array.dtype == np.uint8


def test_load_pair_keeps_alignment(tmp_path):
    # A photo and sketch of different sizes with a mark at the same relative place.
    photo = np.zeros((300, 250, 3), dtype=np.uint8)
    photo[:150, :125] = 255  # top-left quarter
    sketch = np.zeros((250, 200), dtype=np.uint8)
    sketch[:125, :100] = 255
    Image.fromarray(photo).save(tmp_path / "p.png")
    Image.fromarray(sketch).save(tmp_path / "s.png")
    p, s = load_pair(tmp_path / "p.png", tmp_path / "s.png")
    assert np.abs(p.astype(int) - s.astype(int)).mean() < 3


def test_build_cache_shapes_dtypes_and_order(tmp_path):
    make_fake_fs2k(tmp_path)
    records = read_annotations(tmp_path / "anno_train.json")[::-1]  # not file order
    out_dir = tmp_path / "cache"
    summary = build_cache(tmp_path, records, out_dir, "train", size=32)

    photos = np.load(out_dir / "train_photos.npy")
    sketches = np.load(out_dir / "train_sketches.npy")
    styles = np.load(out_dir / "train_styles.npy")
    assert photos.shape == (12, 32, 32, 3) and photos.dtype == np.uint8
    assert sketches.shape == (12, 32, 32, 3) and sketches.dtype == np.uint8
    assert styles.shape == (12,) and styles.dtype == np.int64
    assert styles.tolist() == [r["style"] for r in records]

    # Row k must belong to records[k]: compare with loading that pair directly.
    for k, record in enumerate(records):
        photo, sketch = load_pair(*resolve_pair(tmp_path, record["name"]), size=32)
        assert np.array_equal(photos[k], photo)
        assert np.array_equal(sketches[k], sketch)

    assert summary["photo_sizes"] == {(250, 250), (250, 300)}
    assert summary["sketch_sizes"] == {(200, 250)}
    assert summary["grey_sketches"] == 12  # all fake sketches are grey


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    make_fake_fs2k(tmp_path / "data" / "FS2K" / "FS2K")  # nested layout
    config = {
        "seed": 42,
        "image_size": 32,
        "data": {"fs2k_dir": "data/FS2K", "manifests_dir": "manifests"},
        "val_fraction": {"fs2k": 0.15},
    }
    (tmp_path / "configs").mkdir()
    with open(tmp_path / "configs" / "base.yaml", "w", encoding="utf-8") as f:
        json.dump(config, f)  # JSON is valid YAML
    monkeypatch.setattr(prepare_fs2k, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare_fs2k"])

    prepare_fs2k.main()
    output = capsys.readouterr().out
    assert "one folder down" in output

    with open(tmp_path / "manifests" / "fs2k_split.json", "r", encoding="utf-8") as f:
        split = json.load(f)
    assert len(split["train"]) == 9 and len(split["val"]) == 3 and len(split["test"]) == 6
    for part in ["train", "val", "test"]:
        n = len(split[part])
        assert np.load(tmp_path / "data" / "fs2k_128" / f"{part}_photos.npy").shape == (n, 32, 32, 3)

    # Running again reuses the saved split.
    prepare_fs2k.main()
    assert "Reusing existing split" in capsys.readouterr().out

    # A different saved split is refused.
    split["val"], split["train"] = split["train"][:3], split["val"] + split["train"][3:]
    with open(tmp_path / "manifests" / "fs2k_split.json", "w", encoding="utf-8") as f:
        json.dump(split, f)
    with pytest.raises(ValueError):
        prepare_fs2k.main()


# ---------- FS2KDataset ----------

def write_cache(cache_dir, photos, sketches, styles, split="train"):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{split}_photos.npy", photos)
    np.save(cache_dir / f"{split}_sketches.npy", sketches)
    np.save(cache_dir / f"{split}_styles.npy", np.array(styles, dtype=np.int64))


def random_images(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (n, 128, 128, 3), dtype=np.uint8)


def test_dataset_shapes_range_and_style(tmp_path):
    photos = random_images(3)
    photos[0] = 0
    photos[1] = 255
    write_cache(tmp_path, photos, random_images(3, seed=1), [0, 1, 2])
    dataset = FS2KDataset(tmp_path, "train")
    assert len(dataset) == 3

    photo, sketch, style = dataset[0]
    assert photo.shape == (3, 128, 128) and sketch.shape == (3, 128, 128)
    assert photo.dtype == torch.float32
    assert torch.all(photo == -1.0)
    assert torch.all(dataset[1][0] == 1.0)
    assert [dataset[i][2] for i in range(3)] == [0, 1, 2]
    for i in range(3):
        for t in dataset[i][:2]:
            assert t.min() >= -1.0 and t.max() <= 1.0


def test_to_unit_range():
    t = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.allclose(to_unit_range(t), torch.tensor([0.0, 0.5, 1.0]))


def test_augmentation_is_identical_for_photo_and_sketch(tmp_path):
    photos = random_images(2)
    write_cache(tmp_path, photos, photos.copy(), [0, 1])  # sketch == photo
    plain = FS2KDataset(tmp_path, "train", augment=False)
    augmented = FS2KDataset(tmp_path, "train", augment=True)

    torch.manual_seed(0)
    changed = 0
    for _ in range(50):
        photo, sketch, _ = augmented[0]
        assert photo.shape == (3, 128, 128)
        assert torch.equal(photo, sketch)  # same flip and same crop
        assert photo.min() >= -1.0 and photo.max() <= 1.0
        if not torch.equal(photo, plain[0][0]):
            changed += 1
    assert changed > 0  # augmentation really does something
