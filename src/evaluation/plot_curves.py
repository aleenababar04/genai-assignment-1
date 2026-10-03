"""Plot training and validation curves of a final run, downloaded from W&B.

Run:  python -m src.evaluation.plot_curves --run final-task1_udae
Writes report/figures/<run name without "final-">_curves.png

Needs WANDB_API_KEY (in .env or the environment).
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

from src.utils.config import PROJECT_ROOT
from src.utils.tracking import DEFAULT_WANDB_PROJECT  # also loads .env

# Panels to draw for each kind of run: (title, [(metric key, label), ...]).
PANELS = {
    "restoration": [
        ("loss", [("train/loss", "train loss")]),
        ("validation objective", [("val/objective", "val objective")]),
        ("validation quality", [("val/psnr", "PSNR (dB)")]),
        ("validation SSIM", [("val/ssim", "SSIM")]),
    ],
    "classifier": [
        ("cross-entropy", [("train/loss", "train"), ("val/loss", "validation")]),
        ("accuracy", [("train/accuracy", "train"), ("val/accuracy", "validation")]),
        ("validation macro F1", [("val/macro_f1", "macro F1")]),
    ],
    "moe": [
        ("loss parts", [("train/loss", "total"), ("train/ce", "CE"), ("train/balance", "balance")]),
        ("validation objective", [("val/objective", "objective")]),
        ("gate", [("val/gate_accuracy", "gate accuracy"), ("val/min_branch_weight", "min branch weight")]),
        ("mean branch weights", [("val/mean_weight_identity", "identity"), ("val/mean_weight_salt_pepper", "salt-pepper"),
                                 ("val/mean_weight_blur", "blur"), ("val/mean_weight_occlusion", "occlusion")]),
    ],
    "gan": [
        ("discriminator", [("train/d_real", "D real"), ("train/d_fake", "D fake")]),
        ("generator", [("train/g_adv", "G adversarial")]),
        ("reconstruction", [("train/g_l1", "G L1")]),
        ("validation", [("val/ssim", "SSIM"), ("val/objective", "objective")]),
    ],
}


def kind_of(run_name):
    if "classifier" in run_name:
        return "classifier"
    if "moe" in run_name:
        return "moe"
    if "cgan" in run_name:
        return "gan"
    return "restoration"


def main():
    parser = argparse.ArgumentParser(description="Plot the curves of a W&B run.")
    parser.add_argument("--run", required=True, help="W&B run name, e.g. final-task1_udae")
    args = parser.parse_args()

    api = wandb.Api()
    project = os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    runs = [r for r in api.runs(f"{api.default_entity}/{project}") if r.name == args.run]
    if not runs:
        raise SystemExit(f"no run named {args.run} in project {project}")
    run = sorted(runs, key=lambda r: r.created_at)[-1]  # newest if repeated
    history = run.scan_history()  # every logged row, as a list of dicts (no pandas needed)
    rows = [row for row in history if row.get("epoch") is not None]

    panels = PANELS[kind_of(args.run)]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 2.8))
    for ax, (title, series) in zip(axes, panels):
        for key, label in series:
            points = [(row["epoch"], row[key]) for row in rows if isinstance(row.get(key), (int, float))]
            if points:
                ax.plot([p[0] for p in points], [p[1] for p in points], label=label, linewidth=1.2)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=7)
        if len(series) > 1:
            ax.legend(fontsize=6)
    fig.tight_layout()

    out_path = PROJECT_ROOT / "report" / "figures" / f"{args.run.removeprefix('final-')}_curves.png"
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
