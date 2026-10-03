"""Analysis of the gating network of the soft mixture of experts (Task 3).

For every image the gate gives four weights w = [w0, w1, w2, w3], one per
branch in BRANCH_NAMES, and the weights of one image sum to 1. Branch k is
the expert for true label k (identity <-> clean, label 0).

This file reports:
  - the mean weights for every (true condition, severity) group,
  - a routing heatmap and a weight-distribution figure,
  - examples where one expert dominates and where weights are spread out,
  - whether any expert is inactive or dominates unrelated inputs.
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.corruptions import CONDITION_NAMES
from src.evaluation.metrics import SEVERITY_ORDER

# Branch k of the mixture is the expert for true label k.
BRANCH_NAMES = ["identity", "salt_pepper", "blur", "occlusion"]

# An image counts as "one expert dominates" if its largest weight is at least this.
DOMINANT_THRESHOLD = 0.8

# Added inside the logarithm so that log(0) never happens.
ENTROPY_EPS = 1e-12


# ---------------------------------------------------------------------------
# Gate weights over a whole dataset
# ---------------------------------------------------------------------------

def collect_weights(model, dataset, device, batch_size: int = 128, num_workers: int = 0) -> np.ndarray:
    """Gate weights of every item of `dataset`, shape (N, 4).

    The model's forward returns (x_hat, weights, logits); only the weights
    are kept. The caller must put the model in eval mode first. Row i
    belongs to dataset.entries[i].
    """
    # shuffle=False keeps the results aligned with dataset.entries.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_weights = []

    with torch.no_grad():
        for corrupted, _, _ in loader:
            weights = model(corrupted.to(device))[1]
            all_weights.append(weights.cpu())

    return torch.cat(all_weights).numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def mean_weights_by_group(entries, weights) -> list:
    """Mean gate weight of each branch for every (condition, severity) group.

    Rows are ordered clean, salt_pepper, blur, occlusion and, inside each
    condition, none, low, medium, high, sampled; empty groups are skipped.
    "top_branch" is the branch with the largest mean weight in the group.
    """
    weights = np.asarray(weights, dtype=np.float64)
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
            means = weights[mask].mean(axis=0)
            row = {"condition": condition, "severity": severity, "count": count}
            for k, name in enumerate(BRANCH_NAMES):
                row[name] = float(means[k])
            row["top_branch"] = BRANCH_NAMES[int(means.argmax())]
            rows.append(row)
    return rows


def routing_entropy(weights) -> np.ndarray:
    """Entropy of each row of weights in nats, shape (N,).

    H = -sum_k w_k * log(w_k). It is 0 when one expert gets all the weight
    and ln(4) ~ 1.386 when the four weights are equal.
    """
    weights = np.asarray(weights, dtype=np.float64)
    return -(weights * np.log(weights + ENTROPY_EPS)).sum(axis=1)


def _pick_spread(scores, entries, n):
    """Indices of the n highest scores, covering different conditions first.

    First pass: the best item of each condition (best conditions first).
    Second pass: fill the remaining places with the best items left overall.
    """
    order = list(np.argsort(-np.asarray(scores), kind="stable"))
    chosen = []
    seen_conditions = set()
    for index in order:
        condition = entries[index]["condition"]
        if condition not in seen_conditions:
            seen_conditions.add(condition)
            chosen.append(int(index))
    chosen = chosen[:n]
    for index in order:
        if len(chosen) >= n:
            break
        if int(index) not in chosen:
            chosen.append(int(index))
    return chosen


def pick_dominant_and_distributed(weights, entries, n: int = 4):
    """Example indices: (dominant, distributed), each a list of at most n ints.

    dominant    : items with the largest top weight (one expert does the work).
    distributed : items with the highest routing entropy (weights spread
                  across several experts).
    Both lists take one item from each true condition first, so the
    examples are not all of the same corruption type.
    """
    weights = np.asarray(weights, dtype=np.float64)
    dominant = _pick_spread(weights.max(axis=1), entries, n)
    distributed = _pick_spread(routing_entropy(weights), entries, n)
    return dominant, distributed


def expert_health(entries, weights, inactive_threshold: float = 0.05, dominance_threshold: float = 0.5) -> dict:
    """Check every expert for inactivity and for dominating unrelated inputs.

    Per branch k (branch k is the expert for true label k):
      mean_weight           : mean w_k over all images.
      top1_share            : fraction of images where w_k is the largest weight.
      mean_weight_on_own    : mean w_k over images whose true label is k
                              (NaN if there are none).
      mean_weight_on_others : mean w_k over images whose true label is not k
                              (NaN if there are none).
      inactive              : the gate has practically stopped using this
                              expert: it is the top expert for fewer than 1% of
                              images AND its mean weight is below
                              `inactive_threshold`.
      dominates_unrelated   : on images that belong to OTHER experts, this
                              expert still gets more than `dominance_threshold`
                              of the weight on average, i.e. it takes over
                              inputs it was not meant for.
    Summary keys: "inactive_branches" and "dominating_branches" list the
    names of the flagged branches.
    """
    weights = np.asarray(weights, dtype=np.float64)
    labels = np.array([entry["label"] for entry in entries])
    top1 = weights.argmax(axis=1)

    health = {"inactive_branches": [], "dominating_branches": []}
    for k, name in enumerate(BRANCH_NAMES):
        own = labels == k
        mean_weight = float(weights[:, k].mean())
        top1_share = float((top1 == k).mean())
        on_own = float(weights[own, k].mean()) if own.any() else float("nan")
        on_others = float(weights[~own, k].mean()) if (~own).any() else float("nan")

        inactive = top1_share < 0.01 and mean_weight < inactive_threshold
        # A NaN comparison is False, so "no other images" never counts as dominating.
        dominates = on_others > dominance_threshold

        health[name] = {
            "mean_weight": mean_weight,
            "top1_share": top1_share,
            "mean_weight_on_own": on_own,
            "mean_weight_on_others": on_others,
            "inactive": bool(inactive),
            "dominates_unrelated": bool(dominates),
        }
        if inactive:
            health["inactive_branches"].append(name)
        if dominates:
            health["dominating_branches"].append(name)
    return health


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def save_routing_heatmap(rows, path, title=None):
    """Heatmap of mean gate weights: one row per (condition, severity) group, one column per branch."""
    matrix = np.array([[row[name] for name in BRANCH_NAMES] for row in rows])
    row_labels = [f"{row['condition']} / {row['severity']}" for row in rows]

    fig, ax = plt.subplots(figsize=(6, 0.45 * len(rows) + 1.4))
    heat = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(BRANCH_NAMES)))
    ax.set_yticks(np.arange(len(rows)))
    ax.set_xticklabels(BRANCH_NAMES, fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("expert branch")
    ax.set_ylabel("true condition / severity")

    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            # viridis is dark for small values, so use white text there.
            color = "black" if value > 0.5 else "white"
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)

    if title is not None:
        ax.set_title(title, fontsize=10)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04, label="mean gate weight")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_weight_distribution(entries, weights, path):
    """Box plots of the four gate weights, one panel per true condition.

    Unlike the heatmap (means only), this shows how much the weights vary
    between images of the same condition.
    """
    weights = np.asarray(weights, dtype=np.float64)
    conditions = np.array([entry["condition"] for entry in entries])

    fig, axes = plt.subplots(1, len(CONDITION_NAMES), figsize=(13, 3.6), sharey=True)
    for ax, condition in zip(axes, CONDITION_NAMES):
        mask = conditions == condition
        if mask.any():
            # One box per branch: the weights of that branch over this condition's images.
            ax.boxplot([weights[mask, k] for k in range(len(BRANCH_NAMES))], showfliers=True)
        ax.set_xticks(np.arange(1, len(BRANCH_NAMES) + 1))
        ax.set_xticklabels(BRANCH_NAMES, fontsize=7, rotation=30)
        ax.set_title(f"true: {condition} (n={int(mask.sum())})", fontsize=9)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("gate weight")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Saving results
# ---------------------------------------------------------------------------

def write_csv(rows, columns, path):
    """Write a list of dicts as a CSV file with the given column order."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


HEALTH_COLUMNS = [
    "branch", "mean_weight", "top1_share", "mean_weight_on_own",
    "mean_weight_on_others", "inactive", "dominates_unrelated",
]


def save_health_csv(health, path):
    """One row per branch from `expert_health`."""
    rows = [{"branch": name, **health[name]} for name in BRANCH_NAMES]
    write_csv(rows, HEALTH_COLUMNS, path)


def save_group_csv(rows, path):
    """One row per (condition, severity) group from `mean_weights_by_group`."""
    write_csv(rows, ["condition", "severity", "count", *BRANCH_NAMES, "top_branch"], path)
