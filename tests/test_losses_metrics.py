"""Tests for the restoration loss and the evaluation metrics."""

import math

import numpy as np
import pytest
import torch

from src.evaluation.metrics import (
    evaluate_restoration,
    l1_per_image,
    mean_metrics,
    psnr_per_image,
    ssim_per_image,
    summarise_by_condition,
)
from src.training.losses import restoration_loss

SIZE = 32  # small images keep the tests fast (SSIM needs at least 11 pixels)


def random_images(batch, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, 3, SIZE, SIZE, generator=generator)


# ---------------------------------------------------------------------------
# restoration_loss
# ---------------------------------------------------------------------------

def test_loss_of_identical_images_is_zero():
    images = random_images(2)
    loss, l1, ssim_value = restoration_loss(images, images.clone(), alpha=0.8)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)
    assert l1.item() == pytest.approx(0.0, abs=1e-7)
    assert ssim_value.item() == pytest.approx(1.0, abs=1e-5)


def test_loss_outputs_are_scalars():
    loss, l1, ssim_value = restoration_loss(random_images(2, 0), random_images(2, 1), alpha=0.8)
    assert loss.shape == () and l1.shape == () and ssim_value.shape == ()


def test_alpha_one_is_pure_l1_and_alpha_zero_is_pure_ssim():
    pred, target = random_images(2, 0), random_images(2, 1)

    loss, l1, ssim_value = restoration_loss(pred, target, alpha=1.0)
    assert loss.item() == pytest.approx(l1.item(), abs=1e-6)

    loss, l1, ssim_value = restoration_loss(pred, target, alpha=0.0)
    assert loss.item() == pytest.approx(1.0 - ssim_value.item(), abs=1e-6)


def test_loss_mixes_the_two_terms():
    pred, target = random_images(2, 0), random_images(2, 1)
    loss, l1, ssim_value = restoration_loss(pred, target, alpha=0.3)
    expected = 0.3 * l1.item() + 0.7 * (1.0 - ssim_value.item())
    assert loss.item() == pytest.approx(expected, abs=1e-6)


def test_loss_is_differentiable():
    pred = random_images(2, 0).requires_grad_(True)
    target = random_images(2, 1)
    loss, _, _ = restoration_loss(pred, target, alpha=0.8)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.shape == pred.shape
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum().item() > 0


def test_noisier_prediction_has_larger_loss():
    target = random_images(2, 0)
    noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(1))
    slightly_noisy = (target + 0.02 * noise).clamp(0, 1)
    very_noisy = (target + 0.20 * noise).clamp(0, 1)
    small_loss, _, _ = restoration_loss(slightly_noisy, target, alpha=0.8)
    large_loss, _, _ = restoration_loss(very_noisy, target, alpha=0.8)
    assert large_loss.item() > small_loss.item()


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------

def test_per_image_shapes():
    pred, target = random_images(4, 0), random_images(4, 1)
    assert l1_per_image(pred, target).shape == (4,)
    assert psnr_per_image(pred, target).shape == (4,)
    assert ssim_per_image(pred, target).shape == (4,)


def test_identical_images_give_perfect_scores():
    images = random_images(3)
    assert torch.allclose(l1_per_image(images, images), torch.zeros(3))
    assert torch.allclose(psnr_per_image(images, images), torch.full((3,), 100.0), atol=1e-3)
    assert torch.allclose(ssim_per_image(images, images), torch.ones(3), atol=1e-5)
    assert torch.isfinite(psnr_per_image(images, images)).all()


def test_constant_offset_gives_known_l1_and_psnr():
    offset = 0.1
    target = torch.full((2, 3, SIZE, SIZE), 0.5)
    pred = target + offset
    # MSE = offset^2, so PSNR = 10 * log10(1 / offset^2) = 20 * log10(1 / offset).
    expected_psnr = 20.0 * math.log10(1.0 / offset)
    assert torch.allclose(l1_per_image(pred, target), torch.full((2,), offset), atol=1e-6)
    assert torch.allclose(psnr_per_image(pred, target), torch.full((2,), expected_psnr), atol=1e-3)


def test_per_image_values_follow_batch_order():
    target = torch.full((3, 3, SIZE, SIZE), 0.5)
    offsets = [0.0, 0.1, 0.3]  # image 0 is perfect, image 2 is the worst
    pred = target.clone()
    for i, offset in enumerate(offsets):
        pred[i] += offset

    l1 = l1_per_image(pred, target)
    psnr = psnr_per_image(pred, target)
    assert torch.allclose(l1, torch.tensor(offsets), atol=1e-6)
    assert psnr[0] > psnr[1] > psnr[2]

    # SSIM needs structure, so use noise of growing strength for it.
    target = random_images(3, 0)
    noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(1))
    strengths = torch.tensor([0.0, 0.05, 0.3]).view(3, 1, 1, 1)
    ssim_values = ssim_per_image((target + strengths * noise).clamp(0, 1), target)
    assert ssim_values[0] > ssim_values[1] > ssim_values[2]


# ---------------------------------------------------------------------------
# evaluate_restoration
# ---------------------------------------------------------------------------

class FakeDataset:
    """Seven items; item i has a clean image of 0.2 and an input of 0.2 + 0.05 * i."""

    def __init__(self, length=7):
        self.entries = [{"id": i} for i in range(length)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        clean = torch.full((3, SIZE, SIZE), 0.2)
        corrupted = clean + 0.05 * index
        return corrupted, clean, 0


def test_evaluate_identity_matches_input_baseline():
    dataset = FakeDataset()
    metrics = evaluate_restoration(lambda x: x, dataset, torch.device("cpu"), batch_size=3)

    assert set(metrics) == {"l1", "psnr", "ssim", "input_psnr", "input_ssim"}
    for values in metrics.values():
        assert isinstance(values, np.ndarray)
        assert len(values) == len(dataset)

    # Doing nothing gives exactly the baseline scores.
    np.testing.assert_allclose(metrics["psnr"], metrics["input_psnr"])
    np.testing.assert_allclose(metrics["ssim"], metrics["input_ssim"])


def test_evaluate_keeps_dataset_order_across_batches():
    dataset = FakeDataset()
    metrics = evaluate_restoration(lambda x: x, dataset, torch.device("cpu"), batch_size=3)
    # Item i was built with an error of 0.05 * i (3 batches: 3 + 3 + 1 items).
    expected_l1 = np.array([0.05 * i for i in range(len(dataset))])
    np.testing.assert_allclose(metrics["l1"], expected_l1, atol=1e-6)


def test_evaluate_clamps_restored_images():
    dataset = FakeDataset()
    device = torch.device("cpu")

    # Far too bright: clamped to 1.0, so the error against 0.2 is 0.8.
    metrics = evaluate_restoration(lambda x: x + 5.0, dataset, device, batch_size=3)
    np.testing.assert_allclose(metrics["l1"], np.full(len(dataset), 0.8), atol=1e-6)

    # Far too dark: clamped to 0.0, so the error against 0.2 is 0.2.
    metrics = evaluate_restoration(lambda x: x - 5.0, dataset, device, batch_size=3)
    np.testing.assert_allclose(metrics["l1"], np.full(len(dataset), 0.2), atol=1e-6)


# ---------------------------------------------------------------------------
# summarise_by_condition and mean_metrics
# ---------------------------------------------------------------------------

def make_summary_inputs():
    """Seven hand-made entries, deliberately not in table order."""
    groups = [
        ("occlusion", "high"),
        ("blur", "low"),
        ("clean", "none"),
        ("salt_pepper", "medium"),
        ("blur", "low"),
        ("salt_pepper", "low"),
        ("blur", "high"),
    ]
    entries = [{"condition": condition, "severity": severity} for condition, severity in groups]
    metrics = {
        "l1": np.array([0.7, 0.2, 0.0, 0.4, 0.4, 0.3, 0.6]),
        "psnr": np.array([10.0, 30.0, 100.0, 20.0, 34.0, 25.0, 15.0]),
        "ssim": np.array([0.3, 0.8, 1.0, 0.6, 0.9, 0.7, 0.5]),
        "input_psnr": np.array([8.0, 28.0, 100.0, 12.0, 30.0, 18.0, 13.0]),
        "input_ssim": np.array([0.2, 0.7, 1.0, 0.3, 0.8, 0.4, 0.4]),
    }
    return entries, metrics


def test_summary_row_order_and_counts():
    entries, metrics = make_summary_inputs()
    rows = summarise_by_condition(entries, metrics)

    order = [(row["condition"], row["severity"]) for row in rows]
    assert order == [
        ("clean", "none"),
        ("salt_pepper", "low"),
        ("salt_pepper", "medium"),
        ("blur", "low"),
        ("blur", "high"),
        ("occlusion", "high"),
        ("corrupted", "all"),
        ("all", "all"),
    ]
    assert [row["count"] for row in rows] == [1, 1, 1, 2, 1, 1, 6, 7]


def test_summary_corrupted_row_excludes_clean():
    entries, metrics = make_summary_inputs()
    corrupted = summarise_by_condition(entries, metrics)[-2]
    keep = [i for i, entry in enumerate(entries) if entry["condition"] != "clean"]
    for name, values in metrics.items():
        assert corrupted[name] == pytest.approx(values[keep].mean())


def test_summary_group_means():
    entries, metrics = make_summary_inputs()
    rows = summarise_by_condition(entries, metrics)

    blur_low = rows[3]  # entries 1 and 4
    assert blur_low["l1"] == pytest.approx(0.3)
    assert blur_low["psnr"] == pytest.approx(32.0)
    assert blur_low["ssim"] == pytest.approx(0.85)
    assert blur_low["input_psnr"] == pytest.approx(29.0)
    assert blur_low["input_ssim"] == pytest.approx(0.75)

    clean = rows[0]  # entry 2 only
    assert clean["psnr"] == pytest.approx(100.0)
    assert isinstance(clean["psnr"], float)


def test_summary_last_row_is_overall_mean():
    entries, metrics = make_summary_inputs()
    overall = summarise_by_condition(entries, metrics)[-1]
    for name, values in metrics.items():
        assert overall[name] == pytest.approx(values.mean())


def test_summary_puts_sampled_after_fixed_severities():
    entries = [
        {"condition": "blur", "severity": "sampled"},
        {"condition": "blur", "severity": "high"},
        {"condition": "clean", "severity": "none"},
    ]
    metrics = {"l1": np.array([0.1, 0.2, 0.3])}
    rows = summarise_by_condition(entries, metrics)
    order = [(row["condition"], row["severity"]) for row in rows]
    assert order == [
        ("clean", "none"), ("blur", "high"), ("blur", "sampled"), ("corrupted", "all"), ("all", "all"),
    ]


def test_mean_metrics():
    means = mean_metrics({"l1": np.array([0.1, 0.3]), "psnr": np.array([20.0, 30.0])})
    assert means == {"l1": pytest.approx(0.2), "psnr": pytest.approx(25.0)}
    assert all(isinstance(value, float) for value in means.values())
