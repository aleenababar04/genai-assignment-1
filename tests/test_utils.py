"""Tests for the shared utilities in src/utils."""

import random

import numpy as np
import torch

from src.utils import tracking
from src.utils.config import load_config, save_config
from src.utils.seed import set_seed


def test_set_seed_is_reproducible():
    set_seed(42)
    torch_a = torch.rand(5)
    numpy_a = np.random.rand(5)
    python_a = random.random()

    set_seed(42)
    torch_b = torch.rand(5)
    numpy_b = np.random.rand(5)
    python_b = random.random()

    assert torch.equal(torch_a, torch_b)
    assert np.array_equal(numpy_a, numpy_b)
    assert python_a == python_b


def test_config_round_trip(tmp_path):
    cfg = {"seed": 42, "image_size": 128, "data": {"pet_dir": "data/oxford-iiit-pet"}}
    path = tmp_path / "nested" / "config.yaml"  # parent folder does not exist yet

    save_config(cfg, path)

    assert path.exists()
    assert load_config(path) == cfg


def test_get_study_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "STUDIES_DIR", tmp_path / "optuna_studies")

    storage = tracking.get_study_storage("my_study")

    assert storage.startswith("sqlite:///")
    assert "my_study" in storage
    assert "\\" not in storage
    assert (tmp_path / "optuna_studies").is_dir()


def test_create_study_is_reloadable(tmp_path, monkeypatch):
    monkeypatch.setattr(tracking, "STUDIES_DIR", tmp_path / "optuna_studies")

    study = tracking.create_study("reload_test")
    study.optimize(lambda trial: trial.suggest_float("x", -1.0, 1.0) ** 2, n_trials=3)

    # Creating it again must load the existing study, not fail or start empty.
    reloaded = tracking.create_study("reload_test")

    assert (tmp_path / "optuna_studies" / "reload_test.db").exists()
    assert len(reloaded.trials) == 3
