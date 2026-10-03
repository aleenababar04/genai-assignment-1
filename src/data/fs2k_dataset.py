"""PyTorch dataset for the FS2K face-to-sketch task (Task 4).

Each item is (photo, sketch, style):
  photo, sketch : float tensors of shape (3, 128, 128) with values in [-1, 1]
                  (sketches are stored as 3-channel RGB, like the photos)
  style         : 0, 1 or 2, the sketch style
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

JITTER_SIZE = 143  # pix2pix: resize to 143x143, then take a random 128x128 crop


def to_tensor(image):
    """uint8 (H, W, 3) array -> float (3, H, W) tensor in [-1, 1]."""
    t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return t * 2.0 - 1.0


def to_unit_range(t):
    """[-1, 1] -> [0, 1], e.g. before saving a tensor as an image."""
    return ((t + 1.0) / 2.0).clamp(0.0, 1.0)


def paired_augment(photo, sketch):
    """Random flip and random crop, applied IDENTICALLY to photo and sketch.

    The photo and its sketch must stay aligned pixel for pixel, otherwise the
    generator would learn to move the face around. So we stack both into one
    tensor and draw the random numbers once: the same flip and the same crop
    position are used for both images. Only torch's random generator is used,
    so DataLoader workers get different (but seedable) random numbers.
    """
    size = photo.shape[-1]
    pair = torch.stack([photo, sketch])  # (2, 3, H, W)

    # Random horizontal flip with probability 0.5.
    if torch.rand(1).item() < 0.5:
        pair = torch.flip(pair, dims=[3])

    # Resize both to 143x143, then cut out the same random 128x128 window.
    pair = F.interpolate(pair, size=(JITTER_SIZE, JITTER_SIZE), mode="bilinear", align_corners=False)
    top = torch.randint(0, JITTER_SIZE - size + 1, (1,)).item()
    left = torch.randint(0, JITTER_SIZE - size + 1, (1,)).item()
    pair = pair[:, :, top:top + size, left:left + size]

    return pair[0], pair[1]


class FS2KDataset(Dataset):
    """Photo/sketch pairs of one split, loaded from the .npy files written by
    prepare_fs2k.py. Set augment=True for training only."""

    def __init__(self, cache_dir, split, augment=False):
        cache_dir = Path(cache_dir)
        self.photos = np.load(cache_dir / f"{split}_photos.npy")
        self.sketches = np.load(cache_dir / f"{split}_sketches.npy")
        self.styles = np.load(cache_dir / f"{split}_styles.npy")
        self.augment = augment

    def __len__(self):
        return len(self.styles)

    def __getitem__(self, index):
        photo = to_tensor(self.photos[index])
        sketch = to_tensor(self.sketches[index])
        if self.augment:
            photo, sketch = paired_augment(photo, sketch)
        return photo, sketch, int(self.styles[index])
