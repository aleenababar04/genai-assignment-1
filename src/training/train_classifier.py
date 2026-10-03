"""Task 2a: train the corruption classifier.

  python -m src.training.train_classifier --mode optuna   # hyperparameter search
  python -m src.training.train_classifier --mode final    # full training with the best settings

Every training batch contains exactly the same number of clean, salt-and-pepper,
blur and occlusion images (see src/data/balanced.py).
"""

import argparse

import optuna
import torch
import torch.nn.functional as F
import wandb

from src.data.balanced import make_balanced_loader
from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, load_images
from src.evaluation.classification import (
    classification_summary,
    normalized_confusion,
    predict_dataset,
    save_confusion_figure,
)
from src.models.autoencoder import count_parameters
from src.models.classifier import CHANNEL_PRESETS, CorruptionClassifier
from src.utils.config import PROJECT_ROOT, get_device, load_config, save_config
from src.utils.seed import set_seed
from src.utils.tracking import create_study, init_wandb, log_checkpoint_artifact

TASK_CONFIG = "configs/task2_classifier.yaml"
BEST_CONFIG = "configs/task2_classifier_best.yaml"


def build_classifier(params):
    """Create the classifier described by a hyperparameter dict."""
    return CorruptionClassifier(channels=CHANNEL_PRESETS[params["channels"]], dropout=params["dropout"])


def load_val_set(cfg):
    """The fixed validation manifest: every validation image in all four conditions."""
    cache_dir = PROJECT_ROOT / cfg["data"]["pet_cache_dir"]
    entries = load_manifest(PROJECT_ROOT / cfg["data"]["manifests_dir"] / "pet_val_manifest.jsonl")
    return PetManifestDataset(load_images(cache_dir, "val"), entries)


def train_one_epoch(model, loader, optimizer, device, max_batches=None):
    """One pass over balanced batches. Returns mean cross-entropy and accuracy."""
    model.train()
    total_loss, correct, seen, num_batches = 0.0, 0, 0, 0
    for corrupted, _, labels in loader:
        corrupted, labels = corrupted.to(device), labels.to(device)

        logits = model(corrupted)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break
    return {"loss": total_loss / num_batches, "accuracy": correct / seen}


def validate(model, val_set, device):
    """Accuracy, macro scores and cross-entropy on the validation manifest."""
    model.eval()
    out = predict_dataset(model, val_set, device)
    stats = classification_summary(out["labels"], out["preds"])
    # Cross-entropy from the stored probabilities: mean of -log p(true class).
    true_probs = out["probs"][range(len(out["labels"])), out["labels"]]
    stats["loss"] = float(-torch.log(torch.as_tensor(true_probs).clamp_min(1e-12)).mean())
    return stats, out


def fit(cfg, task, params, epochs, device, run=None, trial=None, checkpoint_path=None, max_batches=None):
    """Train one classifier and return the validation stats of its best epoch."""
    set_seed(cfg["seed"])
    cache_dir = PROJECT_ROOT / cfg["data"]["pet_cache_dir"]
    train_loader = make_balanced_loader(
        load_images(cache_dir, "train"), params["batch_size"], cfg["num_workers"], cfg["seed"],
        pin_memory=(device.type == "cuda"),
    )
    val_set = load_val_set(cfg)

    model = build_classifier(params).to(device)
    # AdamW applies weight decay directly to the weights (decoupled from the gradient).
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best = None
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, device, max_batches)
        val_stats, val_out = validate(model, val_set, device)
        scheduler.step()

        print(
            f"epoch {epoch:3d}/{epochs}  train loss {train_stats['loss']:.4f}  "
            f"val acc {val_stats['accuracy']:.4f}  macro F1 {val_stats['macro_f1']:.4f}  "
            f"val loss {val_stats['loss']:.4f}"
        )
        if run is not None:
            log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            log.update({f"train/{k}": v for k, v in train_stats.items()})
            log.update({f"val/{k}": v for k, v in val_stats.items() if k != "per_class"})
            run.log(log)

        # Best epoch: highest macro F1; ties are broken by the lower validation loss.
        is_better = best is None or (val_stats["macro_f1"], -val_stats["loss"]) > (best["macro_f1"], -best["loss"])
        if is_better:
            best = dict(val_stats, epoch=epoch, confusion=normalized_confusion(val_out["labels"], val_out["preds"]))
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_args": {"channels": CHANNEL_PRESETS[params["channels"]], "dropout": params["dropout"]},
                        "params": params,
                        "epoch": epoch,
                        "val": {k: v for k, v in val_stats.items() if k != "per_class"},
                    },
                    checkpoint_path,
                )

        if trial is not None:
            trial.report(val_stats["macro_f1"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best


def suggest_params(trial, space):
    """Draw one hyperparameter set from the search space in the config."""
    return {
        "lr": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "channels": trial.suggest_categorical("channels", space["channels"]),
        "dropout": trial.suggest_float("dropout", space["dropout"][0], space["dropout"][1]),
        "weight_decay": trial.suggest_float(
            "weight_decay", space["weight_decay"][0], space["weight_decay"][1], log=True
        ),
    }


def run_optuna(cfg, task, device, n_trials, epochs, max_batches):
    """Hyperparameter search maximising validation macro F1."""
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=task["optuna"]["pruner_startup_trials"],
        n_warmup_steps=task["optuna"]["pruner_warmup_epochs"],
    )
    study = create_study(task["study_name"], direction="maximize", pruner=pruner)

    def objective(trial):
        params = suggest_params(trial, task["search_space"])
        run = init_wandb(task["wandb_group"], dict(params, epochs=epochs),
                         run_name=f"trial-{trial.number:03d}", tags=["optuna-trial"])
        try:
            best = fit(cfg, task, params, epochs, device, run=run, trial=trial, max_batches=max_batches)
        finally:
            run.finish()
        for name in ("accuracy", "macro_precision", "macro_recall", "loss", "epoch"):
            trial.set_user_attr(name, best[name])
        return best["macro_f1"]

    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"trials: {len(study.trials)} total, {len(completed)} completed, {len(pruned)} pruned")
    print(f"best trial {study.best_trial.number}: macro F1 {study.best_value:.5f}")
    print(f"best params: {study.best_params}")
    save_config(dict(study.best_params), PROJECT_ROOT / BEST_CONFIG)
    print(f"saved {BEST_CONFIG}")


def run_final(cfg, task, device, epochs, max_batches):
    """Full training with the best settings found by Optuna (or the defaults)."""
    best_path = PROJECT_ROOT / BEST_CONFIG
    if best_path.exists():
        params = load_config(best_path)
        print(f"using Optuna result from {BEST_CONFIG}")
    else:
        params = dict(task["defaults"])
        print("no Optuna result found, using the defaults from the task config")

    name = task["checkpoint_name"]
    checkpoint_path = PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt"
    print(f"parameters: {count_parameters(build_classifier(params)):,}")

    run = init_wandb(task["wandb_group"], dict(params, epochs=epochs), run_name=f"final-{name}", tags=["final"])
    try:
        best = fit(cfg, task, params, epochs, device, run=run,
                   checkpoint_path=checkpoint_path, max_batches=max_batches)
        # Confusion matrix of the best epoch on the validation set.
        figure_path = PROJECT_ROOT / "report" / "figures" / f"{name}_val_confusion.png"
        save_confusion_figure(best["confusion"], figure_path, "Validation confusion matrix (row-normalised)")
        run.log({"val/confusion": wandb.Image(str(figure_path))})
        summary = {k: v for k, v in best.items() if k not in ("per_class", "confusion")}
        for key, value in summary.items():
            run.summary[f"best/{key}"] = value
        log_checkpoint_artifact(run, checkpoint_path, name, metadata=dict(params, **summary))
    finally:
        run.finish()
    print(f"best epoch {best['epoch']}: accuracy {best['accuracy']:.4f}  macro F1 {best['macro_f1']:.4f}")
    print(f"saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Task 2 corruption classifier.")
    parser.add_argument("--mode", choices=["optuna", "final"], required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--trials", type=int, help="override the number of Optuna trials")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--num-workers", type=int, help="override num_workers from the base config")
    parser.add_argument("--max-batches", type=int, help="limit batches per epoch (quick tests only)")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    task = load_config(PROJECT_ROOT / TASK_CONFIG)
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    device = get_device()
    print(f"device: {device}")

    if args.mode == "optuna":
        run_optuna(cfg, task, device,
                   n_trials=args.trials or task["optuna"]["n_trials"],
                   epochs=args.epochs or task["optuna"]["epochs_per_trial"],
                   max_batches=args.max_batches)
    else:
        run_final(cfg, task, device, epochs=args.epochs or task["final"]["epochs"], max_batches=args.max_batches)


if __name__ == "__main__":
    main()
