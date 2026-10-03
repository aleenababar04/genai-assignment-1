"""Tests for the gating-network analysis (Task 3). Small hand-made data only."""

import csv
import math

import numpy as np
import torch
from torch import nn

from src.evaluation.routing_analysis import (
    BRANCH_NAMES,
    collect_weights,
    expert_health,
    mean_weights_by_group,
    pick_dominant_and_distributed,
    routing_entropy,
    save_group_csv,
    save_health_csv,
    save_routing_heatmap,
    save_weight_distribution,
)

LABELS = {"clean": 0, "salt_pepper": 1, "blur": 2, "occlusion": 3}


def make_entry(condition, severity):
    return {"condition": condition, "label": LABELS[condition], "severity": severity}


def one_hot(k):
    row = [0.0, 0.0, 0.0, 0.0]
    row[k] = 1.0
    return row


# ---------------------------------------------------------------------------
# mean_weights_by_group
# ---------------------------------------------------------------------------

def test_mean_weights_by_group_means_counts_and_order():
    # Deliberately shuffled so the function has to sort the groups.
    entries = [
        make_entry("blur", "high"),
        make_entry("clean", "none"),
        make_entry("blur", "low"),
        make_entry("blur", "high"),
        make_entry("clean", "none"),
    ]
    weights = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.4, 0.0, 0.6, 0.0],
        [0.0, 0.2, 0.6, 0.2],
        [0.6, 0.2, 0.2, 0.0],
    ])
    rows = mean_weights_by_group(entries, weights)

    assert [(r["condition"], r["severity"]) for r in rows] == [
        ("clean", "none"), ("blur", "low"), ("blur", "high"),
    ]
    clean, blur_low, blur_high = rows
    assert clean["count"] == 2
    assert math.isclose(clean["identity"], 0.8)
    assert math.isclose(clean["salt_pepper"], 0.1)
    assert clean["top_branch"] == "identity"
    assert blur_low["count"] == 1 and blur_low["top_branch"] == "blur"
    assert blur_high["count"] == 2
    assert math.isclose(blur_high["blur"], 0.8)
    assert math.isclose(blur_high["occlusion"], 0.1)
    assert isinstance(blur_high["blur"], float)


# ---------------------------------------------------------------------------
# routing_entropy
# ---------------------------------------------------------------------------

def test_routing_entropy_one_hot_and_uniform():
    weights = np.array([one_hot(2), [0.25, 0.25, 0.25, 0.25]])
    entropy = routing_entropy(weights)
    assert entropy.shape == (2,)
    assert abs(entropy[0]) < 1e-6
    assert math.isclose(entropy[1], math.log(4), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# pick_dominant_and_distributed
# ---------------------------------------------------------------------------

def test_pick_dominant_and_distributed_spreads_across_conditions():
    entries = [
        make_entry("blur", "high"),         # 0: very confident
        make_entry("blur", "low"),          # 1: very confident too (same condition)
        make_entry("occlusion", "high"),    # 2: confident
        make_entry("clean", "none"),        # 3: uniform (most spread)
        make_entry("salt_pepper", "low"),   # 4: fairly spread
        make_entry("salt_pepper", "high"),  # 5: uniform too (same condition as 4)
    ]
    weights = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.01, 0.0, 0.99, 0.0],
        [0.05, 0.0, 0.05, 0.9],
        [0.25, 0.25, 0.25, 0.25],
        [0.4, 0.3, 0.3, 0.0],
        [0.25, 0.25, 0.25, 0.25],
    ])
    dominant, distributed = pick_dominant_and_distributed(weights, entries, n=2)

    # The two most confident items are both blur; the second pick must be another condition.
    assert dominant == [0, 2]
    # Items 3 and 5 tie as most spread, item 4 is next; one per condition first.
    assert len(distributed) == 2
    assert {entries[i]["condition"] for i in distributed} == {"clean", "salt_pepper"}
    assert all(isinstance(i, int) for i in dominant + distributed)


def test_pick_returns_at_most_n_and_fills_from_same_condition():
    entries = [make_entry("blur", "high") for _ in range(3)]
    weights = np.array([one_hot(2), [0.1, 0.1, 0.7, 0.1], [0.3, 0.2, 0.3, 0.2]])
    dominant, distributed = pick_dominant_and_distributed(weights, entries, n=2)
    assert dominant == [0, 1]
    assert distributed == [2, 1]
    dominant, _ = pick_dominant_and_distributed(weights, entries, n=10)
    assert sorted(dominant) == [0, 1, 2]


# ---------------------------------------------------------------------------
# expert_health
# ---------------------------------------------------------------------------

def balanced_entries():
    return [make_entry(c, "none" if c == "clean" else "high") for c in LABELS for _ in range(2)]


def test_expert_health_healthy_case():
    entries = balanced_entries()
    weights = np.array([[0.1, 0.1, 0.1, 0.1] for _ in entries])
    for i, entry in enumerate(entries):
        weights[i, entry["label"]] = 0.7
    health = expert_health(entries, weights)

    assert health["inactive_branches"] == []
    assert health["dominating_branches"] == []
    for name in BRANCH_NAMES:
        assert math.isclose(health[name]["mean_weight_on_own"], 0.7)
        assert math.isclose(health[name]["mean_weight_on_others"], 0.1)
        assert math.isclose(health[name]["top1_share"], 0.25)
        assert math.isclose(health[name]["mean_weight"], 0.25)


def test_expert_health_flags_inactive_expert():
    entries = balanced_entries()
    # The occlusion expert never gets any weight; occlusion images go to identity.
    weights = []
    for entry in entries:
        label = entry["label"]
        weights.append(one_hot(0) if label == 3 else one_hot(label))
    health = expert_health(entries, np.array(weights))

    assert health["inactive_branches"] == ["occlusion"]
    assert health["occlusion"]["inactive"] is True
    assert health["occlusion"]["top1_share"] == 0.0
    assert health["identity"]["inactive"] is False


def test_expert_health_flags_dominating_expert():
    entries = balanced_entries()
    # Blur takes 0.7 of the weight on every image, whatever its condition.
    weights = np.array([[0.1, 0.1, 0.7, 0.1] for _ in entries])
    health = expert_health(entries, weights)

    assert health["dominating_branches"] == ["blur"]
    assert health["blur"]["dominates_unrelated"] is True
    assert math.isclose(health["blur"]["mean_weight_on_others"], 0.7)
    assert health["identity"]["dominates_unrelated"] is False


# ---------------------------------------------------------------------------
# collect_weights
# ---------------------------------------------------------------------------

class FakeGateModel(nn.Module):
    """Weights depend on the input: softmax of the first pixel of each channel plus a zero."""

    def forward(self, x):
        features = x[:, :, 0, 0]  # (B, 3)
        logits = torch.cat([features, torch.zeros(x.shape[0], 1)], dim=1)
        return x, torch.softmax(logits, dim=1), logits


class FakeDataset:
    def __init__(self, count):
        self.entries = [make_entry("clean", "none") for _ in range(count)]
        # Image i is filled with the value i / 10, so its weights are unique.
        self.images = [torch.full((3, 4, 4), i / 10.0) for i in range(count)]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        return self.images[index], self.images[index], 0


def test_collect_weights_shape_and_order():
    dataset = FakeDataset(7)
    model = FakeGateModel().eval()
    weights = collect_weights(model, dataset, torch.device("cpu"), batch_size=3)

    assert weights.shape == (7, 4)
    for i in range(7):
        _, expected, _ = model(dataset.images[i].unsqueeze(0))
        assert np.allclose(weights[i], expected[0].numpy(), atol=1e-6)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Figures and CSV files
# ---------------------------------------------------------------------------

def sample_data():
    entries = balanced_entries()
    rng = np.random.default_rng(0)
    weights = rng.dirichlet(np.ones(4), size=len(entries))
    return entries, weights


def test_figures_are_written(tmp_path):
    entries, weights = sample_data()
    rows = mean_weights_by_group(entries, weights)

    heatmap = tmp_path / "sub" / "heatmap.png"
    save_routing_heatmap(rows, heatmap, title="routing")
    assert heatmap.exists() and heatmap.stat().st_size > 0

    distribution = tmp_path / "distribution.png"
    save_weight_distribution(entries, weights, distribution)
    assert distribution.exists() and distribution.stat().st_size > 0


def read_header(path):
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def test_csv_writers_headers(tmp_path):
    entries, weights = sample_data()

    group_path = tmp_path / "groups.csv"
    save_group_csv(mean_weights_by_group(entries, weights), group_path)
    assert read_header(group_path) == [
        "condition", "severity", "count", "identity", "salt_pepper", "blur", "occlusion", "top_branch",
    ]

    health_path = tmp_path / "health.csv"
    save_health_csv(expert_health(entries, weights), health_path)
    assert read_header(health_path) == [
        "branch", "mean_weight", "top1_share", "mean_weight_on_own",
        "mean_weight_on_others", "inactive", "dominates_unrelated",
    ]
    with open(health_path, newline="", encoding="utf-8") as f:
        assert len(list(csv.reader(f))) == 1 + len(BRANCH_NAMES)
