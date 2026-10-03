"""Task 2: evaluate the classifier and the hard-routed restoration system.

On the fixed test manifest this script produces:
  - classifier results: accuracy, macro precision/recall/F1, per-class scores,
    accuracy per severity and the normalised confusion matrix;
  - restoration results in oracle-routing and predicted-routing mode;
  - the misrouted images, with how much each lost compared with oracle routing.

Run:  python -m src.evaluation.eval_task2
"""

import argparse
import csv

import numpy as np
import torch

from src.data.corruptions import CONDITION_NAMES
from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, load_images
from src.evaluation.classification import (
    accuracy_by_severity,
    classification_summary,
    normalized_confusion,
    predict_dataset,
    save_confusion_figure,
    save_per_class_csv,
    save_severity_csv,
)
from src.evaluation.eval_udae import load_autoencoder, save_table
from src.evaluation.figures import describe, pick_representative, save_example_grid, save_metric_bars
from src.evaluation.metrics import evaluate_restoration, summarise_by_condition
from src.models.classifier import CorruptionClassifier
from src.models.hard_router import EXPERT_NAMES, HardRoutedRestorer, make_restore_fn
from src.training.train_specialists import SPECIALISTS
from src.utils.config import PROJECT_ROOT, get_device, load_config
from src.utils.tracking import init_wandb

CLASSIFIER_CONFIG = "configs/task2_classifier.yaml"
SPECIALIST_CONFIG = "configs/task2_specialists.yaml"
NUM_MISROUTED_EXAMPLES = 6


def load_classifier(checkpoint_path, device):
    """Rebuild the classifier from a checkpoint written by train_classifier.py."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = CorruptionClassifier(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def load_router(cfg, device):
    """Classifier plus the three specialists, joined into one hard-routed system."""
    checkpoints_dir = PROJECT_ROOT / cfg["checkpoints_dir"]
    classifier_task = load_config(PROJECT_ROOT / CLASSIFIER_CONFIG)
    specialist_task = load_config(PROJECT_ROOT / SPECIALIST_CONFIG)

    classifier, _ = load_classifier(checkpoints_dir / f"{classifier_task['checkpoint_name']}.pt", device)
    experts = {}
    for name in SPECIALISTS:
        path = checkpoints_dir / f"{specialist_task['checkpoint_prefix']}_{name}.pt"
        experts[name], _ = load_autoencoder(path, device)
    router = HardRoutedRestorer(classifier, experts["salt_pepper"], experts["blur"], experts["occlusion"])
    return router.to(device).eval()


def misrouting_table(entries, labels, preds, oracle, predicted):
    """For every (true class, chosen route) pair with errors: count and mean PSNR loss."""
    rows = []
    for true in range(4):
        for pred in range(4):
            if true == pred:
                continue
            mask = (labels == true) & (preds == pred)
            if mask.sum() == 0:
                continue
            rows.append({
                "true": CONDITION_NAMES[true],
                "routed_to": EXPERT_NAMES[pred],
                "count": int(mask.sum()),
                "oracle_psnr": float(oracle["psnr"][mask].mean()),
                "predicted_psnr": float(predicted["psnr"][mask].mean()),
                # For a clean image the oracle output is the input itself (capped
                # 100 dB PSNR), so the SSIM loss is the more meaningful number there.
                "psnr_loss": float((oracle["psnr"][mask] - predicted["psnr"][mask]).mean()),
                "ssim_loss": float((oracle["ssim"][mask] - predicted["ssim"][mask]).mean()),
            })
    return rows


def misrouted_examples(router, dataset, labels, preds, oracle, predicted, device):
    """The misrouted images that lost the most PSNR compared with oracle routing."""
    wrong = np.flatnonzero(labels != preds)
    worst = wrong[np.argsort(oracle["psnr"][wrong] - predicted["psnr"][wrong])[::-1]]
    examples = []
    for index in worst[:NUM_MISROUTED_EXAMPLES]:
        corrupted, clean, _ = dataset[int(index)]
        with torch.no_grad():
            restored, _, _ = router(corrupted.unsqueeze(0).to(device))
        entry = dataset.entries[int(index)]
        examples.append({
            "clean": clean,
            "corrupted": corrupted,
            "restored": restored[0].cpu(),
            "label": (f"{entry['condition']} / {entry['severity']}\n"
                      f"routed to {EXPERT_NAMES[preds[index]]}\n"
                      f"PSNR {predicted['psnr'][index]:.1f} (oracle {oracle['psnr'][index]:.1f})"),
        })
    return examples


def print_comparison(oracle_rows, predicted_rows):
    print(f"{'condition':<12}{'severity':<9}{'input':>8}{'oracle':>8}{'predicted':>11}   (PSNR)")
    for o, p in zip(oracle_rows, predicted_rows):
        print(f"{o['condition']:<12}{o['severity']:<9}{o['input_psnr']:>8.2f}{o['psnr']:>8.2f}{p['psnr']:>11.2f}")


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        if not rows:
            f.write("no misrouted images\n")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate the Task 2 hard-routing system on the test manifest.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--limit-images", type=int, help="use only the first N test images (quick tests only)")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    device = get_device()
    router = load_router(cfg, device)

    images = load_images(PROJECT_ROOT / cfg["data"]["pet_cache_dir"], "test")
    entries = load_manifest(PROJECT_ROOT / cfg["data"]["manifests_dir"] / "pet_test_manifest.jsonl")
    if args.limit_images is not None:
        entries = [e for e in entries if e["image_index"] < args.limit_images]
    dataset = PetManifestDataset(images, entries)
    results_dir = PROJECT_ROOT / "report" / "results"
    figures_dir = PROJECT_ROOT / "report" / "figures"

    # 1. Classifier.
    out = predict_dataset(router.classifier, dataset, device)
    labels, preds = out["labels"], out["preds"]
    summary = classification_summary(labels, preds)
    confusion = normalized_confusion(labels, preds)
    severity_rows = accuracy_by_severity(entries, labels, preds)
    print(f"classifier: accuracy {summary['accuracy']:.4f}  macro P {summary['macro_precision']:.4f}  "
          f"macro R {summary['macro_recall']:.4f}  macro F1 {summary['macro_f1']:.4f}")
    save_per_class_csv(summary, results_dir / "task2_classifier_per_class.csv")
    save_severity_csv(severity_rows, results_dir / "task2_classifier_by_severity.csv")
    save_confusion_figure(confusion, figures_dir / "task2_test_confusion.png", "Test confusion matrix (row-normalised)")

    # 2. Restoration with oracle and predicted routing.
    #    A fresh oracle restore_fn is needed for every pass (it walks through the entries).
    oracle = evaluate_restoration(make_restore_fn(router, "oracle", entries), dataset, device)
    predicted = evaluate_restoration(make_restore_fn(router, "predicted"), dataset, device)
    oracle_rows = summarise_by_condition(entries, oracle)
    predicted_rows = summarise_by_condition(entries, predicted)
    print_comparison(oracle_rows, predicted_rows)
    save_table(oracle_rows, results_dir / "task2_oracle_test_metrics.csv")
    save_table(predicted_rows, results_dir / "task2_predicted_test_metrics.csv")
    save_metric_bars(predicted_rows, figures_dir / "task2_predicted_psnr_bars.png", "psnr",
                     "Task 2 (predicted routing): PSNR before and after restoration")

    # 3. Failures caused by classifier errors.
    misrouted = misrouting_table(entries, labels, preds, oracle, predicted)
    write_rows(misrouted, results_dir / "task2_misrouting.csv")
    print(f"misrouted images: {int((labels != preds).sum())} of {len(labels)}")
    for row in misrouted:
        print(f"  {row['true']:<12} -> {row['routed_to']:<12} {row['count']:>6}  "
              f"PSNR loss {row['psnr_loss']:.2f} dB  SSIM loss {row['ssim_loss']:.4f}")
    examples = misrouted_examples(router, dataset, labels, preds, oracle, predicted, device)
    if examples:
        save_example_grid(examples, figures_dir / "task2_misrouted.png")

    # 4. Representative predicted-routing examples.
    chosen = pick_representative(entries, predicted)
    representative = []
    for index in chosen:
        corrupted, clean, _ = dataset[index]
        with torch.no_grad():
            restored, routes, _ = router(corrupted.unsqueeze(0).to(device))
        representative.append({
            "clean": clean, "corrupted": corrupted, "restored": restored[0].cpu(),
            "label": describe(entries[index], predicted, index) + f"\nrouted to {EXPERT_NAMES[int(routes[0])]}",
        })
    save_example_grid(representative, figures_dir / "task2_examples.png")
    print(f"saved tables to {results_dir} and figures to {figures_dir}")

    # 5. Record the evaluation in W&B.
    import wandb

    run = init_wandb("task2-eval", {"test_items": len(dataset)}, run_name="eval-task2", tags=["eval"])
    try:
        for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
            run.summary[f"test/classifier_{key}"] = summary[key]
        run.summary["test/oracle_psnr"] = oracle_rows[-1]["psnr"]
        run.summary["test/predicted_psnr"] = predicted_rows[-1]["psnr"]
        run.summary["test/misrouted"] = int((labels != preds).sum())
        logged = {
            "test/confusion": wandb.Image(str(figures_dir / "task2_test_confusion.png")),
            "test/examples": wandb.Image(str(figures_dir / "task2_examples.png")),
            "test/predicted_metrics": wandb.Table(columns=list(predicted_rows[0].keys()),
                                                  data=[list(r.values()) for r in predicted_rows]),
        }
        if examples:
            logged["test/misrouted"] = wandb.Image(str(figures_dir / "task2_misrouted.png"))
        run.log(logged)
    finally:
        run.finish()


if __name__ == "__main__":
    main()
