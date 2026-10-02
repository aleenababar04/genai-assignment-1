"""PyTorch datasets for the Oxford-IIIT Pet restoration tasks (Tasks 1-3).

Both datasets return (corrupted, clean, label):
  corrupted, clean : float tensors of shape (3, 128, 128) with values in [0, 1]
  label            : 0 clean, 1 salt-and-pepper, 2 blur, 3 occlusion
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data import corruptions as C


def load_split(manifests_dir):
    """Read the fixed train/val/test name lists written by prepare_pet.py."""
    with open(Path(manifests_dir) / "pet_split.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_images(cache_dir, split):
    """Load the cached clean images of one split: uint8 array (N, 128, 128, 3)."""
    return np.load(Path(cache_dir) / f"{split}.npy")


def to_tensor(image):
    """uint8 (H, W, 3) array -> float (3, H, W) tensor in [0, 1]."""
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


class PetTrainDataset(Dataset):
    """Training data: a new random corruption every time an image is loaded.

    `conditions` lists the input conditions to choose from with equal
    probability. The default is all four (Task 1). A specialist in Task 2
    passes a single condition, e.g. conditions=(C.BLUR,).
    """

    def __init__(self, images, conditions=(C.CLEAN, C.SALT_PEPPER, C.BLUR, C.OCCLUSION)):
        self.images = images
        self.conditions = tuple(conditions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        clean = to_tensor(self.images[index])
        corrupted, label, _ = C.corrupt_random(clean, self.conditions)
        return corrupted, clean, label


class PetManifestDataset(Dataset):
    """Validation / test data: corruptions replayed from a fixed manifest.

    One item per manifest entry, so an image appears once per stored
    corruption. `self.entries[i]` describes item i (type, severity, params).
    """

    def __init__(self, images, entries):
        self.images = images
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        clean = to_tensor(self.images[entry["image_index"]])
        # Salt-and-pepper needs random numbers; the stored seed makes them repeatable.
        generator = torch.Generator().manual_seed(entry["seed"])
        corrupted = C.apply_corruption(clean, entry["label"], entry["params"], generator=generator)
        return corrupted, clean, entry["label"]
