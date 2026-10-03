"""Balanced training batches for the corruption classifier (Task 2) and the
mixture-of-experts (Task 3).

Every batch contains EXACTLY the same number of images of each condition
(0 clean, 1 salt-and-pepper, 2 blur, 3 occlusion), so the classifier never
sees one class more often than another and cannot become biased toward it.

How it works:
  1. PetCleanDataset returns only clean images.
  2. The DataLoader groups them into batches (drop_last=True, so every batch
     is full).
  3. balanced_corruption_collate gives the batch the labels 0,1,2,3,0,1,2,3,...,
     shuffles them, and corrupts each image according to its label.

All randomness uses the global torch random number generator, so the
DataLoader's per-worker seeding gives every worker different corruptions.
"""

import torch
from torch.utils.data import DataLoader, Dataset

from src.data import corruptions as C
from src.data.pet_dataset import to_tensor
from src.utils.seed import make_generator, seed_worker

NUM_CONDITIONS = len(C.CONDITION_NAMES)  # 4


class PetCleanDataset(Dataset):
    """Clean images only: uint8 array (N, 128, 128, 3) -> float tensor (3, 128, 128)."""

    def __init__(self, images):
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return to_tensor(self.images[index])  # (3, 128, 128) in [0, 1]


def balanced_corruption_collate(batch):
    """Turn a list of clean images into a perfectly balanced corrupted batch.

    batch: list of B clean tensors, each (3, 128, 128). B must be a multiple of 4.
    Returns (corrupted, clean, labels):
      corrupted, clean : float tensors (B, 3, 128, 128)
      labels           : long tensor (B,)
    """
    batch_size = len(batch)
    if batch_size % NUM_CONDITIONS != 0:
        raise ValueError(
            f"Batch size must be a multiple of {NUM_CONDITIONS} so that every "
            f"condition gets the same number of images; got {batch_size}.")

    # Labels 0,1,2,3,0,1,2,3,... -> exactly B/4 of each condition.
    labels = torch.arange(batch_size) % NUM_CONDITIONS          # (B,)
    # Shuffle them, so the position in the batch says nothing about the label.
    labels = labels[torch.randperm(batch_size)]                 # (B,)

    corrupted = []
    for clean, label in zip(batch, labels.tolist()):
        _, height, width = clean.shape
        params = C.sample_params(label, height, width)
        corrupted.append(C.apply_corruption(clean, label, params))

    corrupted = torch.stack(corrupted)   # (B, 3, 128, 128)
    clean = torch.stack(batch)           # (B, 3, 128, 128)
    return corrupted, clean, labels.long()


def make_balanced_loader(images, batch_size: int, num_workers: int, seed: int,
                         pin_memory: bool = False) -> DataLoader:
    """DataLoader whose every batch is exactly balanced over the 4 conditions.

    drop_last=True drops the last, smaller batch, so every batch is full and
    therefore balanced. `seed` fixes the shuffling order of the images.
    """
    if batch_size % NUM_CONDITIONS != 0:
        raise ValueError(
            f"batch_size must be a multiple of {NUM_CONDITIONS} for balanced "
            f"batches; got {batch_size}.")

    return DataLoader(
        PetCleanDataset(images),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=balanced_corruption_collate,
        worker_init_fn=seed_worker,
        generator=make_generator(seed),
    )
