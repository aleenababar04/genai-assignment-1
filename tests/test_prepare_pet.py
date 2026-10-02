"""Tests for src/data/prepare_pet.py. No network and no real dataset needed."""

import random

import numpy as np
from PIL import Image

from src.data.prepare_pet import build_cache, load_image, make_split, read_image_names


def test_read_image_names_parses_first_column(tmp_path):
    list_file = tmp_path / "trainval.txt"
    list_file.write_text(
        "Abyssinian_100 1 1 1\n"
        "\n"
        "beagle_12 5 2 3\n"
        "   \n"
        "american_bulldog_7 2 2 1\n",
        encoding="utf-8",
    )
    names = read_image_names(list_file)
    assert names == ["Abyssinian_100", "beagle_12", "american_bulldog_7"]


def test_make_split_sizes():
    names = [f"img_{i:04d}" for i in range(100)]
    train, val = make_split(names)
    assert len(train) == 80
    assert len(val) == 20

    # 3680 is the size of the official trainval list.
    names = [f"img_{i:04d}" for i in range(3680)]
    train, val = make_split(names)
    assert len(train) == 2944
    assert len(val) == 736


def test_make_split_is_disjoint_and_complete():
    names = [f"img_{i:04d}" for i in range(100)]
    train, val = make_split(names)
    assert set(train) & set(val) == set()
    assert sorted(train + val) == sorted(names)
    # Each list is returned in alphabetical order.
    assert train == sorted(train)
    assert val == sorted(val)


def test_make_split_ignores_input_order():
    names = [f"img_{i:04d}" for i in range(100)]
    shuffled = list(names)
    random.Random(0).shuffle(shuffled)
    assert shuffled != names
    assert make_split(shuffled) == make_split(names)


def test_make_split_is_repeatable():
    names = [f"img_{i:04d}" for i in range(100)]
    assert make_split(names) == make_split(names)


def test_make_split_changes_with_seed():
    names = [f"img_{i:04d}" for i in range(100)]
    assert make_split(names, seed=42) != make_split(names, seed=7)


def test_load_image_rgb(tmp_path):
    path = tmp_path / "rgb.jpg"
    Image.new("RGB", (160, 160), (200, 30, 30)).save(path)
    array = load_image(path)
    assert array.shape == (128, 128, 3)
    assert array.dtype == np.uint8


def test_load_image_greyscale_jpeg(tmp_path):
    path = tmp_path / "grey.jpg"
    Image.new("L", (160, 160), 120).save(path)
    array = load_image(path)
    assert array.shape == (128, 128, 3)
    assert array.dtype == np.uint8


def test_load_image_rgba_and_palette_png(tmp_path):
    rgba_path = tmp_path / "rgba.png"
    Image.new("RGBA", (160, 160), (10, 200, 10, 128)).save(rgba_path)
    palette_path = tmp_path / "palette.png"
    Image.new("RGB", (160, 160), (10, 10, 200)).convert("P").save(palette_path)

    for path in [rgba_path, palette_path]:
        array = load_image(path)
        assert array.shape == (128, 128, 3)
        assert array.dtype == np.uint8


def test_load_image_non_square(tmp_path):
    path = tmp_path / "wide.jpg"
    Image.new("RGB", (300, 200), (50, 100, 150)).save(path)
    array = load_image(path)
    assert array.shape == (128, 128, 3)
    assert array.dtype == np.uint8


def test_build_cache_shape_dtype_and_order(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    # One plain colour per image, so we can tell the rows apart.
    colours = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255)}
    for name, colour in colours.items():
        Image.new("RGB", (200, 150), colour).save(images_dir / f"{name}.jpg")

    names = ["blue", "red", "green"]  # deliberately not alphabetical
    out_file = tmp_path / "cache" / "train.npy"
    shape = build_cache(names, images_dir, out_file)
    assert shape == (3, 128, 128, 3)

    array = np.load(out_file)
    assert array.shape == (3, 128, 128, 3)
    assert array.dtype == np.uint8

    # Row i must be the image for names[i]. JPEG is lossy, so the colours are
    # close to the originals but not exactly equal.
    for row, name in zip(array, names):
        mean_colour = row.reshape(-1, 3).mean(axis=0)
        assert np.allclose(mean_colour, colours[name], atol=10)
