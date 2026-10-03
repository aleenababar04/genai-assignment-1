"""Tests for the classifier evaluation (Task 2)."""

import csv

import numpy as np
import pytest
import torch
from torch import nn

from src.evaluation.classification import (
    accuracy_by_severity,
    classification_summary,
    normalized_confusion,
    predict_dataset,
    save_confusion_figure,
    save_per_class_csv,
    save_severity_csv,
)

# Hand-made example: 8 images, 2 mistakes (a clean predicted as salt_pepper,
# a blur predicted as occlusion).
LABELS = np.array([0, 0, 0, 1, 1, 2, 2, 3])
PREDS = np.array([0, 0, 1, 1, 1, 2, 3, 3])


# ---------------------------------------------------------------------------
# classification_summary
# ---------------------------------------------------------------------------

def test_summary_matches_hand_computed_values():
    summary = classification_summary(LABELS, PREDS)
    assert summary["accuracy"] == pytest.approx(6 / 8)

    # Per class (precision, recall, f1, support), computed by hand.
    expected = {
        "clean": (1.0, 2 / 3, 0.8, 3),
        "salt_pepper": (2 / 3, 1.0, 0.8, 2),
        "blur": (1.0, 0.5, 2 / 3, 2),
        "occlusion": (0.5, 1.0, 2 / 3, 1),
    }
    assert [row["class"] for row in summary["per_class"]] == list(expected)
    for row in summary["per_class"]:
        precision, recall, f1, support = expected[row["class"]]
        assert row["precision"] == pytest.approx(precision)
        assert row["recall"] == pytest.approx(recall)
        assert row["f1"] == pytest.approx(f1)
        assert row["support"] == support

    assert summary["macro_precision"] == pytest.approx((1 + 2 / 3 + 1 + 0.5) / 4)
    assert summary["macro_recall"] == pytest.approx((2 / 3 + 1 + 0.5 + 1) / 4)
    assert summary["macro_f1"] == pytest.approx((0.8 + 0.8 + 2 / 3 + 2 / 3) / 4)
    assert isinstance(summary["accuracy"], float)


# ---------------------------------------------------------------------------
# normalized_confusion
# ---------------------------------------------------------------------------

def test_confusion_hand_made_case():
    expected = np.array([
        [2 / 3, 1 / 3, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.5, 0.5],
        [0.0, 0.0, 0.0, 1.0],
    ])
    matrix = normalized_confusion(LABELS, PREDS)
    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, expected)
    assert np.allclose(matrix.sum(axis=1), 1.0)


def test_confusion_of_perfect_predictions_is_identity():
    labels = np.array([0, 1, 2, 3, 3, 2])
    assert np.allclose(normalized_confusion(labels, labels), np.eye(4))


def test_confusion_empty_row_stays_zero():
    # No occlusion images at all: the last row has no samples.
    labels = np.array([0, 1, 2, 2])
    preds = np.array([0, 1, 2, 3])
    matrix = normalized_confusion(labels, preds)
    assert np.allclose(matrix[3], 0.0)
    assert np.allclose(matrix[:3].sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# accuracy_by_severity
# ---------------------------------------------------------------------------

def make_entries(pairs):
    return [{"condition": condition, "severity": severity} for condition, severity in pairs]


def test_accuracy_by_severity():
    # Deliberately out of order, to check the row ordering.
    entries = make_entries([
        ("blur", "high"), ("blur", "high"), ("blur", "high"),
        ("clean", "none"), ("clean", "none"),
        ("blur", "low"),
        ("salt_pepper", "medium"),
    ])
    labels = [2, 2, 2, 0, 0, 2, 1]
    preds = [3, 3, 2, 0, 0, 2, 0]
    rows = accuracy_by_severity(entries, labels, preds)

    assert [(r["condition"], r["severity"]) for r in rows] == [
        ("clean", "none"), ("salt_pepper", "medium"), ("blur", "low"), ("blur", "high"),
    ]
    clean, salt, blur_low, blur_high = rows
    assert clean["count"] == 2 and clean["accuracy"] == pytest.approx(1.0)
    assert clean["most_common_wrong"] is None
    assert salt["accuracy"] == pytest.approx(0.0) and salt["most_common_wrong"] == "clean"
    assert blur_low["count"] == 1 and blur_low["most_common_wrong"] is None
    assert blur_high["count"] == 3
    assert blur_high["accuracy"] == pytest.approx(1 / 3)
    assert blur_high["most_common_wrong"] == "occlusion"


# ---------------------------------------------------------------------------
# predict_dataset
# ---------------------------------------------------------------------------

class MeanModel(nn.Module):
    """Fake classifier: predicts class round(3 * image mean)."""

    def forward(self, x):
        mean = x.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1) * 3  # (B, 1)
        classes = torch.arange(4, dtype=x.dtype).unsqueeze(0)  # (1, 4)
        # The closest class gets the largest logit.
        return -10.0 * (classes - mean).abs()


class FakeDataset:
    """Tiny dataset of constant images; image i has value label / 3."""

    def __init__(self, labels):
        self.entries = [{"label": label} for label in labels]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        label = self.entries[index]["label"]
        image = torch.full((3, 8, 8), label / 3)
        return image, image, label


def test_predict_dataset_shapes_and_order():
    labels = [0, 3, 1, 2, 2, 0, 1, 3, 3, 1, 0]
    dataset = FakeDataset(labels)
    # batch_size 4 splits the 11 items into three batches.
    result = predict_dataset(MeanModel().eval(), dataset, torch.device("cpu"), batch_size=4)

    assert result["labels"].shape == (11,)
    assert result["preds"].shape == (11,)
    assert result["probs"].shape == (11, 4)
    assert result["probs"].dtype == np.float32
    assert np.allclose(result["probs"].sum(axis=1), 1.0, atol=1e-5)
    # The fake model is always right, so predictions repeat the labels in order.
    assert result["labels"].tolist() == labels
    assert result["preds"].tolist() == labels


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def test_save_confusion_figure(tmp_path):
    path = tmp_path / "figures" / "confusion.png"
    save_confusion_figure(normalized_confusion(LABELS, PREDS), path, title="test")
    assert path.exists() and path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.reader(f))


def test_csv_writers(tmp_path):
    summary = classification_summary(LABELS, PREDS)
    save_per_class_csv(summary, tmp_path / "per_class.csv")
    lines = read_csv(tmp_path / "per_class.csv")
    assert lines[0] == ["class", "precision", "recall", "f1", "support"]
    assert len(lines) == 1 + 4

    entries = make_entries([("clean", "none"), ("blur", "low"), ("blur", "low")])
    rows = accuracy_by_severity(entries, [0, 2, 2], [0, 2, 1])
    save_severity_csv(rows, tmp_path / "severity.csv")
    lines = read_csv(tmp_path / "severity.csv")
    assert lines[0] == ["condition", "severity", "count", "accuracy", "most_common_wrong"]
    assert len(lines) == 1 + 2
