"""Seeding helpers so every run is reproducible."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch (CPU and GPU) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # does nothing if there is no GPU
    # Always pick the same convolution algorithm, instead of the fastest one.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Seed NumPy and Python `random` inside a DataLoader worker.

    PyTorch gives each worker its own torch seed, but NumPy and `random`
    are not seeded automatically. Pass this as `worker_init_fn`.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42) -> torch.Generator:
    """Return a seeded torch.Generator. Pass it as `generator` to a DataLoader
    so the shuffling order is the same on every run."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
