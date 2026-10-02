"""Task 1: evaluate the trained autoencoder on the fixed test manifest.

Writes a results table per corruption and severity, a figure with twelve
representative examples and a figure with four failure cases.

Run:  python -m src.evaluation.eval_udae
"""

import argparse
import csv

import torch

from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, load_images
from src.evaluation.figures import (
    describe,
    pick_failures,
    pick_representative,
    save_example_grid,
    save_metric_bars,
)
from src.evaluation.metrics import evaluate_restoration, summarise_by_condition
from src.models.autoencoder import ConvAutoencoder
from src.utils.config import PROJECT_ROOT, get_device, load_config
from src.utils.tracking import init_wandb

TASK_CONFIG = "configs/task1_udae.yaml"


def load_autoencoder(checkpoint_path, device):
    """Rebuild the model from a checkpoint written by the training script."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ConvAutoencoder(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def save_table(rows, path):
    """Write the summary rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    print(f"{'condition':<12}{'severity':<9}{'count':>6}{'in PSNR':>9}{'PSNR':>8}{'in SSIM':>9}{'SSIM':>8}{'L1':>8}")
    for r in rows:
        print(
            f"{r['condition']:<12}{r['severity']:<9}{r['count']:>6}{r['input_psnr']:>9.2f}{r['psnr']:>8.2f}"
            f"{r['input_ssim']:>9.4f}{r['ssim']:>8.4f}{r['l1']:>8.4f}"
        )


def build_examples(dataset, model, indices, metrics, device):
    """Collect clean / corrupted / restored tensors for the chosen items."""
    examples = []
    for index in indices:
        corrupted, clean, _ = dataset[index]
        with torch.no_grad():
            restored = model(corrupted.unsqueeze(0).to(device))[0].cpu()
        examples.append({
            "clean": clean,
            "corrupted": corrupted,
            "restored": restored,
            "label": describe(dataset.entries[index], metrics, index),
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Task 1 autoencoder on the test manifest.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", help="default: checkpoints/<checkpoint_name>.pt")
    parser.add_argument("--name", help="prefix of the output files (default: the checkpoint name)")
    parser.add_argument("--limit-images", type=int, help="use only the first N test images (quick tests only)")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    task = load_config(PROJECT_ROOT / TASK_CONFIG)
    device = get_device()

    name = args.name or task["checkpoint_name"]
    checkpoint_path = args.checkpoint or PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt"
    model, checkpoint = load_autoencoder(checkpoint_path, device)

    images = load_images(PROJECT_ROOT / cfg["data"]["pet_cache_dir"], "test")
    entries = load_manifest(PROJECT_ROOT / cfg["data"]["manifests_dir"] / "pet_test_manifest.jsonl")
    if args.limit_images is not None:
        entries = [e for e in entries if e["image_index"] < args.limit_images]
    dataset = PetManifestDataset(images, entries)

    metrics = evaluate_restoration(model, dataset, device)
    rows = summarise_by_condition(entries, metrics)
    print_table(rows)

    results_dir = PROJECT_ROOT / "report" / "results"
    figures_dir = PROJECT_ROOT / "report" / "figures"
    save_table(rows, results_dir / f"{name}_test_metrics.csv")
    save_metric_bars(rows, figures_dir / f"{name}_psnr_bars.png", "psnr", "Task 1: PSNR before and after restoration")
    save_metric_bars(rows, figures_dir / f"{name}_ssim_bars.png", "ssim", "Task 1: SSIM before and after restoration")

    examples = build_examples(dataset, model, pick_representative(entries, metrics), metrics, device)
    save_example_grid(examples, figures_dir / f"{name}_examples.png")
    failures = build_examples(dataset, model, pick_failures(entries, metrics), metrics, device)
    save_example_grid(failures, figures_dir / f"{name}_failures.png")
    print(f"saved table to {results_dir} and figures to {figures_dir}")

    # Record the evaluation in W&B as well.
    import wandb

    run = init_wandb(task["wandb_group"], dict(checkpoint["params"], test_items=len(dataset)),
                     run_name=f"eval-{name}", tags=["eval"])
    try:
        table = wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows])
        run.log({
            "test/metrics": table,
            "test/examples": wandb.Image(str(figures_dir / f"{name}_examples.png")),
            "test/failures": wandb.Image(str(figures_dir / f"{name}_failures.png")),
        })
        for key in ("psnr", "ssim", "l1"):
            run.summary[f"test/{key}"] = rows[-1][key]  # the last row is the overall mean
    finally:
        run.finish()


if __name__ == "__main__":
    main()
