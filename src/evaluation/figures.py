"""Figure helpers shared by the restoration tasks (Tasks 1-3)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

# Error maps use one fixed colour scale so different figures can be compared.
ERROR_MAP_MAX = 0.5


def to_image(tensor):
    """(3, H, W) tensor in [0, 1] -> (H, W, 3) array for matplotlib."""
    return tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def error_map(restored, clean):
    """Absolute error per pixel, averaged over the three colour channels."""
    return (restored - clean).abs().mean(dim=0).detach().cpu().numpy()


def save_example_grid(examples, path, title=None):
    """Save one row per example: clean target, corrupted input, restoration, error map.

    `examples` is a list of dicts with keys "clean", "corrupted", "restored"
    (tensors of shape (3, H, W)) and "label" (text written left of the row).
    """
    column_titles = ["clean target", "corrupted input", "restored output", "absolute error"]
    rows = len(examples)
    fig, axes = plt.subplots(rows, 4, figsize=(8.2, 2.0 * rows + 0.4), squeeze=False)

    for row, example in enumerate(examples):
        panels = [
            to_image(example["clean"]),
            to_image(example["corrupted"]),
            to_image(example["restored"]),
        ]
        for col, panel in enumerate(panels):
            axes[row, col].imshow(panel)
        heat = axes[row, 3].imshow(
            error_map(example["restored"], example["clean"]), cmap="inferno", vmin=0.0, vmax=ERROR_MAP_MAX
        )
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(column_titles[col], fontsize=9)
        axes[row, 0].set_ylabel(example["label"], fontsize=7)

    if title is not None:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.colorbar(heat, ax=axes[:, 3], fraction=0.05, pad=0.03)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_representative(entries, metrics, extra_clean=2):
    """Indices of typical examples: the median-PSNR item of every (condition, severity) group.

    Clean images get `extra_clean` more examples (lower and upper quartile),
    so the default gives 10 + 2 = 12 examples on the test manifest.
    """
    groups = {}
    for index, entry in enumerate(entries):
        groups.setdefault((entry["condition"], entry["severity"]), []).append(index)

    chosen = []
    for (condition, _), indices in groups.items():
        order = sorted(indices, key=lambda i: metrics["psnr"][i])
        chosen.append(order[len(order) // 2])
        if condition == "clean" and extra_clean >= 2:
            chosen.append(order[len(order) // 4])
            chosen.append(order[3 * len(order) // 4])
    return chosen


def pick_failures(entries, metrics):
    """Index of the lowest-SSIM item of each condition: four failure cases."""
    worst = {}
    for index, entry in enumerate(entries):
        condition = entry["condition"]
        if condition not in worst or metrics["ssim"][index] < metrics["ssim"][worst[condition]]:
            worst[condition] = index
    return list(worst.values())


def describe(entry, metrics, index):
    """Row label: condition, severity and the scores of that item."""
    return (
        f"{entry['condition']} / {entry['severity']}\n"
        f"PSNR {metrics['psnr'][index]:.1f}  SSIM {metrics['ssim'][index]:.3f}"
    )


def save_metric_bars(rows, path, metric="psnr", title=None):
    """Bar chart of input vs restored quality for every (condition, severity) row."""
    rows = [r for r in rows if r["condition"] != "all"]
    labels = [f"{r['condition']}\n{r['severity']}" for r in rows]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(x - 0.2, [r[f"input_{metric}"] for r in rows], width=0.4, label="corrupted input")
    ax.bar(x + 0.2, [r[metric] for r in rows], width=0.4, label="restored output")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel(metric.upper())
    if title is not None:
        ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
