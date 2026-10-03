"""Evaluation of the condition classifier (Task 2).

The classifier predicts the input condition of an image:
0 clean, 1 salt_pepper, 2 blur, 3 occlusion (CONDITION_NAMES).
Reported: accuracy, macro precision / recall / F1, per-class scores,
a normalised confusion matrix and accuracy per (condition, severity) group.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from src.data.corruptions import CONDITION_NAMES
from src.evaluation.metrics import SEVERITY_ORDER

# Class indices 0..3, passed to sklearn so all four classes always appear.
CLASS_INDICES = list(range(len(CONDITION_NAMES)))


# ---------------------------------------------------------------------------
# Predictions over a whole dataset
# ---------------------------------------------------------------------------

def predict_dataset(model, dataset, device, batch_size: int = 256, num_workers: int = 0) -> dict:
    """Run the classifier on every item of `dataset`.

    The caller must put the model in eval mode first. Returns NumPy arrays in
    the same order as `dataset.entries`:
      labels : true class of each item, shape (N,)
      preds  : predicted class (argmax of the logits), shape (N,)
      probs  : softmax probabilities, shape (N, 4)
    """
    # shuffle=False keeps the results aligned with dataset.entries.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_labels, all_probs = [], []

    with torch.no_grad():
        for corrupted, _, labels in loader:
            logits = model(corrupted.to(device))
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_labels.append(labels)

    probs = torch.cat(all_probs).numpy().astype(np.float32)
    labels = torch.cat(all_labels).numpy().astype(np.int64)
    # The most probable class is the prediction (same as argmax of the logits).
    preds = probs.argmax(axis=1).astype(np.int64)
    return {"labels": labels, "preds": preds, "probs": probs}


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def classification_summary(labels, preds) -> dict:
    """Accuracy, macro-averaged precision / recall / F1 and per-class scores.

    Macro averaging gives every class the same weight, whatever its size.
    A class that is never predicted gets precision 0 (zero_division=0).
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=CLASS_INDICES, zero_division=0
    )
    per_class = []
    for index, name in enumerate(CONDITION_NAMES):
        per_class.append({
            "class": name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        })
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
    }


def normalized_confusion(labels, preds) -> np.ndarray:
    """4x4 confusion matrix; row = true class, column = predicted class.

    Each row is divided by its total, so entry [i, j] is the fraction of
    class i images predicted as class j. A row with no images stays 0.
    """
    counts = confusion_matrix(labels, preds, labels=CLASS_INDICES).astype(np.float64)
    row_totals = counts.sum(axis=1, keepdims=True)
    # Divide only where the row total is non-zero, to avoid 0 / 0.
    return np.divide(counts, row_totals, out=np.zeros_like(counts), where=row_totals > 0)


def accuracy_by_severity(entries, labels, preds) -> list:
    """Accuracy for each (condition, severity) group of the manifest.

    Item i of `labels` and `preds` belongs to entries[i]. Rows are ordered
    clean, salt_pepper, blur, occlusion and, inside each condition, none,
    low, medium, high, sampled; empty groups are skipped.
    "most_common_wrong" is the class most often predicted by mistake in the
    group (None if every item is correct).
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
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
            correct = preds[mask] == labels[mask]
            wrong_preds = preds[mask][~correct]
            most_common_wrong = None
            if len(wrong_preds) > 0:
                # Count how often each class was wrongly predicted; ties go to the lower index.
                wrong_counts = np.bincount(wrong_preds, minlength=len(CONDITION_NAMES))
                most_common_wrong = CONDITION_NAMES[int(wrong_counts.argmax())]
            rows.append({
                "condition": condition,
                "severity": severity,
                "count": count,
                "accuracy": float(correct.mean()),
                "most_common_wrong": most_common_wrong,
            })
    return rows


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def save_confusion_figure(matrix, path, title=None):
    """Heatmap of a normalised confusion matrix with the value written in each cell."""
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4.4))
    heat = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)

    ticks = np.arange(len(CONDITION_NAMES))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(CONDITION_NAMES, fontsize=8)
    ax.set_yticklabels(CONDITION_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            # White text is easier to read on the dark cells.
            color = "white" if value > 0.5 else "black"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=9)

    if title is not None:
        ax.set_title(title, fontsize=10)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_csv(rows, columns, path):
    """Write a list of dicts as a CSV file with the given column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_per_class_csv(summary, path):
    """One row per class: precision, recall, F1 and support."""
    write_csv(summary["per_class"], ["class", "precision", "recall", "f1", "support"], path)


def save_severity_csv(rows, path):
    """One row per (condition, severity) group from `accuracy_by_severity`."""
    write_csv(rows, ["condition", "severity", "count", "accuracy", "most_common_wrong"], path)
