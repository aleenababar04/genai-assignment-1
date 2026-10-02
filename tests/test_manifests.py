"""Tests for the corruption manifests and the two Pet datasets."""

from collections import Counter

import numpy as np
import torch

from src.data import corruptions as C
from src.data.manifests import (
    build_test_manifest,
    build_val_manifest,
    load_manifest,
    save_manifest,
)
from src.data.pet_dataset import PetManifestDataset, PetTrainDataset, to_tensor

NAMES = [f"pet_{i}" for i in range(5)]


def fake_images(n=5, size=128):
    """Random uint8 images standing in for the cached dataset."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, size, size, 3), dtype=np.uint8)


def test_val_manifest_is_balanced_and_in_range():
    entries = build_val_manifest(NAMES)
    assert len(entries) == 4 * len(NAMES)
    assert Counter(e["label"] for e in entries) == {0: 5, 1: 5, 2: 5, 3: 5}
    assert len({e["seed"] for e in entries}) == len(entries)  # every seed is unique

    for e in entries:
        p = e["params"]
        if e["label"] == C.SALT_PEPPER:
            assert C.SP_PROB_RANGE[0] <= p["prob"] <= C.SP_PROB_RANGE[1]
        elif e["label"] == C.BLUR:
            assert p["kernel_size"] in C.BLUR_KERNEL_SIZES
            assert C.BLUR_SIGMA_RANGE[0] <= p["sigma"] <= C.BLUR_SIGMA_RANGE[1]
        elif e["label"] == C.OCCLUSION:
            assert 1 <= len(p["rects"]) <= 3
            assert C.OCC_COVERAGE_RANGE[0] <= C.coverage_of(p["rects"], 128, 128) <= C.OCC_COVERAGE_RANGE[1]
        else:
            assert p == {}


def test_test_manifest_has_fixed_severities():
    entries = build_test_manifest(NAMES)
    assert len(entries) == 10 * len(NAMES)

    for image_index in range(len(NAMES)):
        group = [e for e in entries if e["image_index"] == image_index]
        assert [e["condition"] for e in group] == (
            ["clean"] + ["salt_pepper"] * 3 + ["blur"] * 3 + ["occlusion"] * 3
        )
        assert [e["severity"] for e in group] == ["none"] + C.SEVERITIES * 3

        assert [e["params"]["prob"] for e in group[1:4]] == [0.03, 0.08, 0.15]
        assert [(e["params"]["kernel_size"], e["params"]["sigma"]) for e in group[4:7]] == [
            (3, 0.7), (5, 1.5), (7, 2.5),
        ]
        for e, (num_rects, target) in zip(group[7:10], [(1, 0.10), (2, 0.20), (3, 0.35)]):
            assert len(e["params"]["rects"]) == num_rects
            assert abs(C.coverage_of(e["params"]["rects"], 128, 128) - target) <= 0.01


def test_manifests_are_deterministic_and_seeds_do_not_collide():
    assert build_val_manifest(NAMES) == build_val_manifest(NAMES)
    assert build_test_manifest(NAMES) == build_test_manifest(NAMES)
    val_seeds = {e["seed"] for e in build_val_manifest(NAMES)}
    test_seeds = {e["seed"] for e in build_test_manifest(NAMES)}
    assert val_seeds.isdisjoint(test_seeds)


def test_manifest_round_trip(tmp_path):
    entries = build_test_manifest(NAMES)
    path = tmp_path / "sub" / "manifest.jsonl"
    save_manifest(entries, path)
    assert load_manifest(path) == entries


def test_manifest_dataset_replays_the_same_corruption():
    images = fake_images()
    entries = build_test_manifest(NAMES)
    dataset = PetManifestDataset(images, entries)
    assert len(dataset) == len(entries)

    for index in range(10):  # the ten entries of the first image
        corrupted_a, clean_a, label_a = dataset[index]
        corrupted_b, clean_b, label_b = dataset[index]
        assert torch.equal(corrupted_a, corrupted_b)  # includes salt-and-pepper noise
        assert torch.equal(clean_a, to_tensor(images[0]))
        assert label_a == label_b == entries[index]["label"]
        assert corrupted_a.shape == (3, 128, 128)
        if label_a == C.CLEAN:
            assert torch.equal(corrupted_a, clean_a)
        else:
            assert not torch.equal(corrupted_a, clean_a)


def test_train_dataset_samples_a_new_corruption_each_load():
    torch.manual_seed(0)
    images = fake_images()
    dataset = PetTrainDataset(images)
    assert len(dataset) == 5

    labels = []
    for _ in range(200):
        corrupted, clean, label = dataset[0]
        assert corrupted.shape == clean.shape == (3, 128, 128)
        assert corrupted.dtype == torch.float32
        assert 0.0 <= corrupted.min() and corrupted.max() <= 1.0
        assert torch.equal(clean, to_tensor(images[0]))
        labels.append(label)
    counts = Counter(labels)
    assert set(counts) == {0, 1, 2, 3}
    assert all(30 <= counts[k] <= 70 for k in counts)  # roughly 50 each


def test_train_dataset_can_be_restricted_to_one_condition():
    dataset = PetTrainDataset(fake_images(), conditions=(C.BLUR,))
    assert {dataset[i % 5][2] for i in range(20)} == {C.BLUR}
