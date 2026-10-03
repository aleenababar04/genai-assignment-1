"""Task 3: evaluate the soft mixture-of-experts and analyse its gate.

On the fixed test manifest this script produces:
  - restoration results per corruption and severity (same table as Tasks 1 and 2);
  - the average routing weights per true corruption and severity, as a CSV
    and a heatmap, plus the spread of the weights per true corruption;
  - examples where one expert dominates and where the weight is shared;
  - a check for inactive experts and experts that dominate unrelated inputs.

Run:  python -m src.evaluation.eval_moe
"""

import argparse

import torch

from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, load_images
from src.evaluation.eval_udae import print_table, save_table
from src.evaluation.figures import pick_failures, pick_representative, save_example_grid, save_metric_bars
from src.evaluation.metrics import evaluate_restoration, summarise_by_condition
from src.evaluation.routing_analysis import (
    expert_health,
    mean_weights_by_group,
    pick_dominant_and_distributed,
    save_group_csv,
    save_health_csv,
    save_routing_heatmap,
    save_weight_distribution,
)
from src.models.autoencoder import ConvAutoencoder
from src.models.classifier import CorruptionClassifier
from src.models.soft_moe import BRANCH_NAMES, SoftMoERestorer
from src.training.train_moe import restore_and_record
from src.utils.config import PROJECT_ROOT, get_device, load_config
from src.utils.tracking import init_wandb

TASK_CONFIG = "configs/task3_moe.yaml"


def load_moe(checkpoint_path, device):
    """Rebuild the mixture from a checkpoint written by train_moe.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    gate = CorruptionClassifier(**checkpoint["gate_args"])
    experts = {name: ConvAutoencoder(**args) for name, args in checkpoint["expert_args"].items()}
    model = SoftMoERestorer(gate, experts["salt_pepper"], experts["blur"], experts["occlusion"],
                            tau=checkpoint["tau"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def weight_text(weights_row):
    """Short text listing the four routing weights of one image."""
    short = {"identity": "id", "salt_pepper": "sp", "blur": "bl", "occlusion": "oc"}
    return " ".join(f"{short[name]} {w:.2f}" for name, w in zip(BRANCH_NAMES, weights_row))


def build_examples(model, dataset, indices, metrics, weights, device):
    """Clean / corrupted / restored tensors for the chosen items, labelled with their weights."""
    examples = []
    for index in indices:
        corrupted, clean, _ = dataset[index]
        with torch.no_grad():
            restored = model(corrupted.unsqueeze(0).to(device))[0][0].cpu()
        entry = dataset.entries[index]
        examples.append({
            "clean": clean,
            "corrupted": corrupted,
            "restored": restored,
            "label": (f"{entry['condition']} / {entry['severity']}\n"
                      f"PSNR {metrics['psnr'][index]:.1f}  SSIM {metrics['ssim'][index]:.3f}\n"
                      f"{weight_text(weights[index])}"),
        })
    return examples


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Task 3 soft mixture-of-experts on the test manifest.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", help="default: checkpoints/<checkpoint_name>.pt")
    parser.add_argument("--limit-images", type=int, help="use only the first N test images (quick tests only)")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    task = load_config(PROJECT_ROOT / TASK_CONFIG)
    device = get_device()
    name = task["checkpoint_name"]
    model, checkpoint = load_moe(args.checkpoint or PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt", device)

    images = load_images(PROJECT_ROOT / cfg["data"]["pet_cache_dir"], "test")
    entries = load_manifest(PROJECT_ROOT / cfg["data"]["manifests_dir"] / "pet_test_manifest.jsonl")
    if args.limit_images is not None:
        entries = [e for e in entries if e["image_index"] < args.limit_images]
    dataset = PetManifestDataset(images, entries)
    results_dir = PROJECT_ROOT / "report" / "results"
    figures_dir = PROJECT_ROOT / "report" / "figures"

    # 1. Restoration quality.
    restore, recorded = restore_and_record(model)  # one pass gives images and weights
    metrics = evaluate_restoration(restore, dataset, device)
    weights = torch.cat(recorded).numpy()
    rows = summarise_by_condition(entries, metrics)
    print_table(rows)
    save_table(rows, results_dir / f"{name}_test_metrics.csv")
    save_metric_bars(rows, figures_dir / f"{name}_psnr_bars.png", "psnr",
                     "Task 3: PSNR before and after restoration")

    # 2. Gate behaviour.
    group_rows = mean_weights_by_group(entries, weights)
    save_group_csv(group_rows, results_dir / f"{name}_mean_weights.csv")
    save_routing_heatmap(group_rows, figures_dir / f"{name}_routing_heatmap.png",
                         "Mean routing weight per true corruption and severity")
    save_weight_distribution(entries, weights, figures_dir / f"{name}_weight_distribution.png")

    health = expert_health(entries, weights)
    save_health_csv(health, results_dir / f"{name}_expert_health.csv")
    print("\nmean routing weights per group:")
    for row in group_rows:
        print(f"  {row['condition']:<12}{row['severity']:<9}" +
              "  ".join(f"{b} {row[b]:.2f}" for b in BRANCH_NAMES) + f"   top: {row['top_branch']}")
    print(f"inactive branches: {health['inactive_branches'] or 'none'}")
    print(f"branches dominating unrelated inputs: {health['dominating_branches'] or 'none'}")

    # 3. Example figures.
    dominant, distributed = pick_dominant_and_distributed(weights, entries)
    save_example_grid(build_examples(model, dataset, dominant, metrics, weights, device),
                      figures_dir / f"{name}_dominant_examples.png", "One expert dominates")
    save_example_grid(build_examples(model, dataset, distributed, metrics, weights, device),
                      figures_dir / f"{name}_distributed_examples.png", "Weight shared across experts")
    save_example_grid(build_examples(model, dataset, pick_representative(entries, metrics), metrics, weights, device),
                      figures_dir / f"{name}_examples.png")
    save_example_grid(build_examples(model, dataset, pick_failures(entries, metrics), metrics, weights, device),
                      figures_dir / f"{name}_failures.png")
    print(f"saved tables to {results_dir} and figures to {figures_dir}")

    # 4. Record the evaluation in W&B.
    import wandb

    run = init_wandb(task["wandb_group"], dict(checkpoint["params"], test_items=len(dataset)),
                     run_name=f"eval-{name}", tags=["eval"])
    try:
        run.log({
            "test/metrics": wandb.Table(columns=list(rows[0].keys()), data=[list(r.values()) for r in rows]),
            "test/mean_weights": wandb.Table(columns=list(group_rows[0].keys()),
                                             data=[list(r.values()) for r in group_rows]),
            "test/routing_heatmap": wandb.Image(str(figures_dir / f"{name}_routing_heatmap.png")),
            "test/weight_distribution": wandb.Image(str(figures_dir / f"{name}_weight_distribution.png")),
            "test/dominant_examples": wandb.Image(str(figures_dir / f"{name}_dominant_examples.png")),
            "test/distributed_examples": wandb.Image(str(figures_dir / f"{name}_distributed_examples.png")),
        })
        for key in ("psnr", "ssim", "l1"):
            run.summary[f"test/{key}"] = rows[-1][key]
            run.summary[f"test/corrupted_{key}"] = rows[-2][key]
        run.summary["test/inactive_branches"] = ", ".join(health["inactive_branches"]) or "none"
        run.summary["test/dominating_branches"] = ", ".join(health["dominating_branches"]) or "none"
    finally:
        run.finish()


if __name__ == "__main__":
    main()
