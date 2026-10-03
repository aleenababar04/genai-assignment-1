"""Task 4: evaluate the face-to-sketch generator on the official FS2K test split.

Produces a metrics table (overall and per style), a grid of test photos with
their true and generated sketches, a style-swap grid (the same photo drawn in
all three styles) and the four worst cases.

Run:  python -m src.evaluation.eval_cgan
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.fs2k_dataset import FS2KDataset, to_unit_range
from src.models.cgan import StyleUNetGenerator
from src.training.train_cgan import fs2k_cache_dir, generate_all, sketch_metrics
from src.utils.config import PROJECT_ROOT, get_device, load_config
from src.utils.tracking import init_wandb

TASK_CONFIG = "configs/task4_cgan.yaml"
NUM_EXAMPLES_PER_STYLE = 3
NUM_FAILURES = 4


def load_generator(checkpoint_path, device):
    """Rebuild the generator from a checkpoint written by train_cgan.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator = StyleUNetGenerator(**checkpoint["generator_args"])
    generator.load_state_dict(checkpoint["generator_state"])
    return generator.to(device).eval(), checkpoint


def as_image(tensor):
    """(C, H, W) in [0, 1] -> (H, W, 3) array for matplotlib (grey sketches are repeated)."""
    image = tensor.clamp(0, 1)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    return image.permute(1, 2, 0).numpy()


def save_grid(rows, column_titles, path, title=None):
    """`rows` is a list of (row label, [tensor, ...]) with one tensor per column."""
    fig, axes = plt.subplots(len(rows), len(column_titles),
                             figsize=(1.9 * len(column_titles), 1.9 * len(rows) + 0.4), squeeze=False)
    for r, (label, images) in enumerate(rows):
        for c, image in enumerate(images):
            axes[r, c].imshow(as_image(image))
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(column_titles[c], fontsize=9)
        axes[r, 0].set_ylabel(label, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Task 4 generator on the FS2K test split.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", help="default: checkpoints/<checkpoint_name>.pt")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    task = load_config(PROJECT_ROOT / TASK_CONFIG)
    device = get_device()
    name = task["checkpoint_name"]
    generator, checkpoint = load_generator(
        args.checkpoint or PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt", device)

    test_set = FS2KDataset(fs2k_cache_dir(cfg), "test")
    fakes, reals, styles = generate_all(generator, test_set, device)
    metrics = sketch_metrics(fakes, reals, styles)

    # 1. Metrics table: overall and per style.
    rows = []
    for label, mask in [("all", np.ones(len(styles), bool))] + [
        (f"style {s + 1}", metrics["style"] == s) for s in range(3)
    ]:
        rows.append({"split": "test", "group": label, "count": int(mask.sum()),
                     **{m: float(metrics[m][mask].mean()) for m in ("l1", "psnr", "ssim")}})
    for r in rows:
        print(f"{r['group']:<8} n={r['count']:<5} L1 {r['l1']:.4f}  PSNR {r['psnr']:.2f}  SSIM {r['ssim']:.4f}")
    results_path = PROJECT_ROOT / "report" / "results" / f"{name}_test_metrics.csv"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figures_dir = PROJECT_ROOT / "report" / "figures"
    photos = lambda i: to_unit_range(test_set[i][0])  # noqa: E731  (photo i in [0, 1])

    # 2. Examples: a few test photos of each style with their true and generated sketches.
    examples = []
    for s in range(3):
        for i in [i for i in range(len(test_set)) if metrics["style"][i] == s][:NUM_EXAMPLES_PER_STYLE]:
            examples.append((f"style {s + 1}\nSSIM {metrics['ssim'][i]:.3f}", [photos(i), reals[i], fakes[i]]))
    save_grid(examples, ["photo", "true sketch", "generated"], figures_dir / f"{name}_examples.png")

    # 3. Style swap: the same photos drawn in all three styles (shows the condition is used).
    with torch.no_grad():
        swap_rows = []
        for i in range(0, min(len(test_set), 5 * 37), 37)[:5]:
            photo = test_set[i][0].unsqueeze(0).to(device)
            outs = [to_unit_range(generator(photo, torch.tensor([s], device=device)).cpu())[0] for s in range(3)]
            swap_rows.append((f"test #{i}", [photos(i)] + outs))
    save_grid(swap_rows, ["photo", "style 1", "style 2", "style 3"], figures_dir / f"{name}_style_swap.png")

    # 4. Failure cases: the lowest SSIM.
    worst = np.argsort(metrics["ssim"])[:NUM_FAILURES]
    failures = [(f"style {metrics['style'][i] + 1}\nSSIM {metrics['ssim'][i]:.3f}", [photos(i), reals[i], fakes[i]])
                for i in worst]
    save_grid(failures, ["photo", "true sketch", "generated"], figures_dir / f"{name}_failures.png")
    print(f"saved {results_path} and figures to {figures_dir}")

    # 5. Record the evaluation in W&B.
    import wandb

    run = init_wandb(task["wandb_group"], dict(checkpoint["params"], test_items=len(test_set)),
                     run_name=f"eval-{name}", tags=["eval"])
    try:
        run.log({
            "test/metrics": wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows]),
            "test/examples": wandb.Image(str(figures_dir / f"{name}_examples.png")),
            "test/style_swap": wandb.Image(str(figures_dir / f"{name}_style_swap.png")),
            "test/failures": wandb.Image(str(figures_dir / f"{name}_failures.png")),
        })
        for key in ("l1", "psnr", "ssim"):
            run.summary[f"test/{key}"] = rows[0][key]
    finally:
        run.finish()


if __name__ == "__main__":
    main()
