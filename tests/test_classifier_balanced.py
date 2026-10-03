"""Tests for the corruption classifier and the balanced batches (CPU, random data)."""

import numpy as np
import pytest
import torch

from src.data.balanced import (PetCleanDataset, balanced_corruption_collate,
                               make_balanced_loader)
from src.models.autoencoder import count_parameters
from src.models.classifier import CHANNEL_PRESETS, CorruptionClassifier


def make_batch(batch_size: int = 2) -> torch.Tensor:
    """A random image batch with values in [0, 1]."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(batch_size, 3, 128, 128, generator=generator)


def make_images(n: int) -> np.ndarray:
    """Random uint8 images (n, 128, 128, 3), like the cached dataset."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(n, 128, 128, 3), dtype=np.uint8)


def clean_list(batch_size: int) -> list:
    """A list of clean (3, 128, 128) tensors, as the DataLoader passes to collate."""
    dataset = PetCleanDataset(make_images(batch_size))
    return [dataset[i] for i in range(batch_size)]


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("preset", list(CHANNEL_PRESETS))
def test_logits_shape(preset):
    model = CorruptionClassifier(channels=CHANNEL_PRESETS[preset])
    logits = model(make_batch(3))
    assert logits.shape == (3, 4)
    assert logits.dtype == torch.float32


def test_no_softmax_applied():
    model = CorruptionClassifier(channels=CHANNEL_PRESETS["small"]).eval()
    with torch.no_grad():
        logits = model(make_batch(4))
    # Softmax outputs would be non-negative and sum to 1 in every row.
    row_sums = logits.sum(dim=1)
    assert not torch.allclose(row_sums, torch.ones_like(row_sums))


def test_parameter_count_grows_with_preset():
    counts = [count_parameters(CorruptionClassifier(channels=CHANNEL_PRESETS[p]))
              for p in ("small", "medium", "large")]
    assert counts[0] < counts[1] < counts[2]


def test_eval_mode_is_deterministic():
    model = CorruptionClassifier(channels=CHANNEL_PRESETS["small"]).eval()
    x = make_batch()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_gradient_reaches_first_conv():
    model = CorruptionClassifier(channels=CHANNEL_PRESETS["small"])
    labels = torch.tensor([0, 3])
    loss = torch.nn.functional.cross_entropy(model(make_batch()), labels)
    loss.backward()
    first_conv = model.block1[0]
    assert first_conv.weight.grad is not None
    assert first_conv.weight.grad.abs().sum().item() > 0


def test_onnx_export_matches_pytorch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    model = CorruptionClassifier(channels=CHANNEL_PRESETS["small"]).eval()
    x = make_batch()
    path = str(tmp_path / "classifier.onnx")

    # dynamo=False selects the legacy (TorchScript) exporter.
    torch.onnx.export(
        model, x, path,
        opset_version=17,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    x3 = make_batch(3)  # a different batch size checks the dynamic batch axis
    with torch.no_grad():
        expected = model(x3).numpy()
    (got,) = session.run(["logits"], {"input": x3.numpy()})
    assert got.shape == (3, 4)
    assert np.abs(got - expected).max() < 1e-4


# ---------------------------------------------------------------------------
# Balanced collate
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("batch_size", [4, 8, 64])
def test_collate_is_exactly_balanced(batch_size):
    torch.manual_seed(0)
    corrupted, clean, labels = balanced_corruption_collate(clean_list(batch_size))
    counts = torch.bincount(labels, minlength=4)
    assert counts.tolist() == [batch_size // 4] * 4


def test_collate_shapes_and_dtypes():
    torch.manual_seed(0)
    corrupted, clean, labels = balanced_corruption_collate(clean_list(8))
    assert corrupted.shape == (8, 3, 128, 128)
    assert clean.shape == (8, 3, 128, 128)
    assert labels.shape == (8,)
    assert corrupted.dtype == torch.float32
    assert clean.dtype == torch.float32
    assert labels.dtype == torch.long


def test_collate_labels_are_shuffled():
    torch.manual_seed(0)
    batch = clean_list(4)
    first_labels = {balanced_corruption_collate(batch)[2][0].item() for _ in range(20)}
    # Without shuffling, position 0 would always be label 0.
    assert len(first_labels) > 1


def test_collate_rejects_non_multiple_of_four():
    with pytest.raises(ValueError):
        balanced_corruption_collate(clean_list(6))


def test_collate_clean_items_unchanged_others_changed():
    torch.manual_seed(0)
    corrupted, clean, labels = balanced_corruption_collate(clean_list(16))
    for i, label in enumerate(labels.tolist()):
        if label == 0:
            assert torch.equal(corrupted[i], clean[i])
        else:
            assert not torch.equal(corrupted[i], clean[i])


# ---------------------------------------------------------------------------
# make_balanced_loader
# ---------------------------------------------------------------------------

def test_loader_every_batch_balanced():
    torch.manual_seed(0)
    n, batch_size = 50, 8
    loader = make_balanced_loader(make_images(n), batch_size=batch_size, num_workers=0, seed=0)
    num_batches = 0
    for corrupted, clean, labels in loader:
        assert corrupted.shape == (batch_size, 3, 128, 128)
        assert torch.bincount(labels, minlength=4).tolist() == [batch_size // 4] * 4
        num_batches += 1
    assert num_batches == n // batch_size
    assert len(loader) == n // batch_size


def test_loader_rejects_non_multiple_of_four():
    with pytest.raises(ValueError):
        make_balanced_loader(make_images(8), batch_size=6, num_workers=0, seed=0)
