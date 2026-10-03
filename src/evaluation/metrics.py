"""Evaluation metrics for image restoration: L1, PSNR and SSIM.

Images are float tensors of shape (B, 3, H, W) with values in [0, 1].
Every metric is computed per image, so the results can later be grouped by
corruption type and severity.
"""

import numpy as np
import torch
from pytorch_msssim import ssim
from torch.utils.data import DataLoader

from src.data.corruptions import CONDITION_NAMES

# Order of the rows in the results table.
SEVERITY_ORDER = ["none", "low", "medium", "high", "sampled"]

# Names of the arrays returned by `evaluate_restoration`.
METRIC_NAMES = ["l1", "psnr", "ssim", "input_psnr", "input_ssim"]

# Smallest allowed MSE, so identical images give 100 dB instead of infinity.
MIN_MSE = 1e-10


# ---------------------------------------------------------------------------
# Per-image metrics
# ---------------------------------------------------------------------------

def l1_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error of each image. Returns shape (B,). Lower is better."""
    # Average over channels, height and width; keep the batch dimension.
    return (pred - target).abs().mean(dim=(1, 2, 3))


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Peak signal-to-noise ratio of each image in dB. Returns shape (B,).

    PSNR = 10 * log10(MAX^2 / MSE) with MAX = 1 because pixels lie in [0, 1].
    Higher is better.
    """
    mse = ((pred - target) ** 2).mean(dim=(1, 2, 3))
    mse = mse.clamp(min=MIN_MSE)
    return 10.0 * torch.log10(1.0 / mse)


def ssim_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Structural similarity of each image. Returns shape (B,). 1 means identical."""
    return ssim(pred, target, data_range=1.0, size_average=False)


# ---------------------------------------------------------------------------
# Evaluation over a whole dataset
# ---------------------------------------------------------------------------

def evaluate_restoration(restore_fn, dataset, device, batch_size: int = 128, num_workers: int = 0) -> dict:
    """Restore every item of `dataset` and measure the result.

    `restore_fn` takes a batch of corrupted images (already on `device`) and
    returns the restored batch. It is a plain callable so that a single
    model, hard routing and the mixture model can all be evaluated here.
    The caller must put any model in eval mode first.

    Returns a dict of NumPy arrays of length len(dataset), in the same order
    as `dataset.entries`:
      l1, psnr, ssim          : restored image vs clean image
      input_psnr, input_ssim  : corrupted input vs clean image (the
                                "do nothing" baseline)
    """
    # shuffle=False keeps the results aligned with dataset.entries.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    batches = {name: [] for name in METRIC_NAMES}

    with torch.no_grad():
        for corrupted, clean, _ in loader:
            corrupted = corrupted.to(device)
            clean = clean.to(device)
            # A network can output values slightly outside the valid range.
            restored = restore_fn(corrupted).clamp(0.0, 1.0)

            batches["l1"].append(l1_per_image(restored, clean).cpu())
            batches["psnr"].append(psnr_per_image(restored, clean).cpu())
            batches["ssim"].append(ssim_per_image(restored, clean).cpu())
            batches["input_psnr"].append(psnr_per_image(corrupted, clean).cpu())
            batches["input_ssim"].append(ssim_per_image(corrupted, clean).cpu())

    # Join the batches into one array per metric.
    return {name: torch.cat(values).numpy() for name, values in batches.items()}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def mean_metrics(metrics: dict) -> dict:
    """Overall mean of each metric array, as Python floats."""
    return {name: float(np.mean(values)) for name, values in metrics.items()}


def summarise_by_condition(entries: list, metrics: dict) -> list:
    """Mean metrics for each (condition, severity) group, plus an overall row.

    `entries` are the manifest entries and `metrics` the arrays returned by
    `evaluate_restoration` (item i of every array belongs to entries[i]).
    Rows are ordered clean, salt_pepper, blur, occlusion and, inside each
    condition, none, low, medium, high, sampled. Groups with no entries are
    skipped. The last two rows are "corrupted / all" (every entry except the
    clean ones) and "all / all" (every entry).

    The "corrupted" row exists because a clean input that is returned
    unchanged scores the capped 100 dB PSNR, which would dominate an
    average that includes it.
    """
    conditions = np.array([entry["condition"] for entry in entries])
    severities = np.array([entry["severity"] for entry in entries])

    rows = []
    for condition in CONDITION_NAMES:
        for severity in SEVERITY_ORDER:
            # Boolean mask that selects the entries of this group.
            mask = (conditions == condition) & (severities == severity)
            count = int(mask.sum())
            if count == 0:
                continue
            row = {"condition": condition, "severity": severity, "count": count}
            for name, values in metrics.items():
                row[name] = float(np.mean(np.asarray(values)[mask]))
            rows.append(row)

    corrupted_mask = conditions != "clean"
    if corrupted_mask.any():
        corrupted = {"condition": "corrupted", "severity": "all", "count": int(corrupted_mask.sum())}
        for name, values in metrics.items():
            corrupted[name] = float(np.mean(np.asarray(values)[corrupted_mask]))
        rows.append(corrupted)

    overall = {"condition": "all", "severity": "all", "count": len(entries)}
    overall.update(mean_metrics(metrics))
    rows.append(overall)
    return rows
