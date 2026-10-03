"""Task 3: train the soft mixture-of-experts.

  python -m src.training.train_moe --mode optuna   # hyperparameter search
  python -m src.training.train_moe --mode final    # full training with the best settings

Nothing starts from random weights: the gate is the trained Task 2
classifier and the experts are the three trained Task 2 specialists.
Training has two stages:
  1. warm-up: experts frozen, only the gate learns;
  2. joint fine-tuning: everything learns, with a smaller learning rate.
"""

import argparse

import numpy as np
import optuna
import torch

from src.data.balanced import make_balanced_loader
from src.data.pet_dataset import load_images
from src.evaluation.eval_task2 import load_classifier
from src.evaluation.eval_udae import load_autoencoder
from src.evaluation.metrics import evaluate_restoration, mean_metrics
from src.models.autoencoder import count_parameters
from src.models.soft_moe import BRANCH_NAMES, SoftMoERestorer, moe_loss
from src.training.train_classifier import load_val_set
from src.training.train_specialists import SPECIALISTS
from src.training.train_udae import log_samples
from src.utils.config import PROJECT_ROOT, get_device, load_config, save_config
from src.utils.seed import set_seed
from src.utils.tracking import create_study, init_wandb, log_checkpoint_artifact

TASK_CONFIG = "configs/task3_moe.yaml"
BEST_CONFIG = "configs/task3_moe_best.yaml"


def build_from_task2(cfg, tau, device):
    """Gate = trained Task 2 classifier, experts = trained Task 2 specialists.

    Returns the model and the architecture arguments needed to rebuild it.
    """
    checkpoints_dir = PROJECT_ROOT / cfg["checkpoints_dir"]
    classifier_task = load_config(PROJECT_ROOT / "configs/task2_classifier.yaml")
    specialist_task = load_config(PROJECT_ROOT / "configs/task2_specialists.yaml")

    gate, gate_checkpoint = load_classifier(checkpoints_dir / f"{classifier_task['checkpoint_name']}.pt", device)
    experts, expert_args = {}, {}
    for name in SPECIALISTS:
        path = checkpoints_dir / f"{specialist_task['checkpoint_prefix']}_{name}.pt"
        experts[name], checkpoint = load_autoencoder(path, device)
        expert_args[name] = checkpoint["model_args"]

    model = SoftMoERestorer(gate, experts["salt_pepper"], experts["blur"], experts["occlusion"], tau=tau)
    architecture = {"gate_args": gate_checkpoint["model_args"], "expert_args": expert_args}
    return model.to(device), architecture


def loss_weights(params):
    """The four lambdas of the joint loss. The SSIM weight is 1 - l1_weight."""
    return {
        "l1_weight": params["l1_weight"],
        "ssim_weight": 1.0 - params["l1_weight"],
        "ce_weight": params["ce_weight"],
        "balance_weight": params["balance_weight"],
    }


def train_one_epoch(model, loader, optimizer, params, device, max_batches=None):
    """One pass over balanced batches. Returns the mean of each loss part."""
    model.train()  # frozen experts stay in eval mode (see SoftMoERestorer.train)
    totals, num_batches = {}, 0
    for corrupted, clean, labels in loader:
        corrupted, clean, labels = corrupted.to(device), clean.to(device), labels.to(device)

        restored, weights, logits = model(corrupted)
        loss, parts = moe_loss(restored, clean, logits, weights, labels, model.tau, **loss_weights(params))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        parts["loss"] = loss.item()
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + value
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break
    return {name: total / num_batches for name, total in totals.items()}


def restore_and_record(model):
    """A restore function for evaluate_restoration that also keeps the routing weights.

    This way one pass over the data gives both the image quality and the
    weights, instead of running all experts twice.
    """
    recorded = []

    def restore(x):
        restored, weights, _ = model(x)
        recorded.append(weights.cpu())
        return restored

    return restore, recorded


def validate(model, val_set, device, objective_alpha):
    """Restoration quality and routing behaviour on the validation manifest."""
    model.eval()
    restore, recorded = restore_and_record(model)
    stats = mean_metrics(evaluate_restoration(restore, val_set, device))
    stats["objective"] = objective_alpha * stats["l1"] + (1 - objective_alpha) * (1 - stats["ssim"])

    weights = torch.cat(recorded).numpy()  # (N, 4), in dataset order
    labels = np.array([entry["label"] for entry in val_set.entries])
    stats["gate_accuracy"] = float((weights.argmax(axis=1) == labels).mean())
    for k, name in enumerate(BRANCH_NAMES):
        stats[f"mean_weight_{name}"] = float(weights[:, k].mean())
    # The smallest average branch weight; close to 0 means a branch is unused (collapse).
    stats["min_branch_weight"] = float(weights.mean(axis=0).min())
    return stats


def fit(cfg, task, params, warmup_epochs, finetune_epochs, device, run=None, trial=None,
        checkpoint_path=None, max_batches=None):
    """Warm-up then joint fine-tuning. Returns the validation stats of the best epoch."""
    set_seed(cfg["seed"])
    model, architecture = build_from_task2(cfg, params["tau"], device)
    loader = make_balanced_loader(
        load_images(PROJECT_ROOT / cfg["data"]["pet_cache_dir"], "train"),
        task["batch_size"], cfg["num_workers"], cfg["seed"], pin_memory=(device.type == "cuda"),
    )
    val_set = load_val_set(cfg)
    sample_indices = [i for i in range(len(val_set)) if val_set.entries[i]["label"] != 0]
    sample_indices = sample_indices[: task["final"]["num_samples"]]

    best = None
    total_epochs = warmup_epochs + finetune_epochs
    for epoch in range(1, total_epochs + 1):
        if epoch == 1:
            # Stage 1: only the gate learns.
            model.freeze_experts()
            optimizer = torch.optim.Adam(model.gate_parameters(), lr=task["warmup"]["lr"])
            scheduler = None
            stage = "warmup"
        if epoch == warmup_epochs + 1:
            # Stage 2: everything learns, with the smaller fine-tuning learning rate.
            model.unfreeze_experts()
            optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=finetune_epochs)
            stage = "finetune"

        train_stats = train_one_epoch(model, loader, optimizer, params, device, max_batches)
        val_stats = validate(model, val_set, device, task["objective_alpha"])
        if scheduler is not None:
            scheduler.step()

        print(
            f"epoch {epoch:3d}/{total_epochs} [{stage}]  train loss {train_stats['loss']:.4f}  "
            f"val objective {val_stats['objective']:.4f}  PSNR {val_stats['psnr']:.2f}  "
            f"gate acc {val_stats['gate_accuracy']:.3f}  min branch weight {val_stats['min_branch_weight']:.3f}"
        )
        if run is not None:
            log = {"epoch": epoch, "stage": 0 if stage == "warmup" else 1, "lr": optimizer.param_groups[0]["lr"]}
            log.update({f"train/{k}": v for k, v in train_stats.items()})
            log.update({f"val/{k}": v for k, v in val_stats.items()})
            run.log(log)
            if trial is None and epoch % task["final"]["sample_every"] == 0:
                log_samples(run, _ImageOnly(model), val_set, sample_indices, device, epoch)

        if best is None or val_stats["objective"] < best["objective"]:
            best = dict(val_stats, epoch=epoch)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"model_state": model.state_dict(), **architecture,
                     "tau": params["tau"], "params": params, "epoch": epoch, "val": val_stats},
                    checkpoint_path,
                )

        if trial is not None:
            trial.report(val_stats["objective"], epoch)
            # Routing collapse: one branch is (almost) never used on a balanced set.
            if stage == "finetune" and val_stats["min_branch_weight"] < task["collapse_threshold"]:
                trial.set_user_attr("pruned_reason", "routing collapse")
                raise optuna.TrialPruned()
            if trial.should_prune():
                trial.set_user_attr("pruned_reason", "median pruner")
                raise optuna.TrialPruned()
    return best


class _ImageOnly(torch.nn.Module):
    """Wraps the mixture so it returns only the restored image (for log_samples)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0]


def suggest_params(trial, space):
    """Draw one hyperparameter set from the search space in the config."""
    return {
        "lr": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "tau": trial.suggest_float("tau", space["tau"][0], space["tau"][1]),
        "ce_weight": trial.suggest_float("ce_weight", space["ce_weight"][0], space["ce_weight"][1], log=True),
        "balance_weight": trial.suggest_float(
            "balance_weight", space["balance_weight"][0], space["balance_weight"][1], log=True
        ),
        "l1_weight": trial.suggest_float("l1_weight", space["l1_weight"][0], space["l1_weight"][1]),
    }


def run_optuna(cfg, task, device, n_trials, warmup_epochs, finetune_epochs, max_batches):
    """Hyperparameter search for the joint fine-tuning stage."""
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=task["optuna"]["pruner_startup_trials"],
        n_warmup_steps=task["optuna"]["pruner_warmup_epochs"],
    )
    study = create_study(task["study_name"], direction="minimize", pruner=pruner)
    # Trial 0 uses the brief's suggested starting values, as a reference point.
    if len(study.trials) == 0:
        study.enqueue_trial(dict(task["defaults"]))

    def objective(trial):
        params = suggest_params(trial, task["search_space"])
        run = init_wandb(task["wandb_group"], dict(params, warmup_epochs=warmup_epochs,
                                                   finetune_epochs=finetune_epochs),
                         run_name=f"trial-{trial.number:03d}", tags=["optuna-trial"])
        try:
            best = fit(cfg, task, params, warmup_epochs, finetune_epochs, device,
                       run=run, trial=trial, max_batches=max_batches)
        finally:
            run.finish()
        for name in ("l1", "psnr", "ssim", "gate_accuracy", "min_branch_weight", "epoch"):
            trial.set_user_attr(name, best[name])
        return best["objective"]

    study.optimize(objective, n_trials=n_trials)

    states = [t.state for t in study.trials]
    collapsed = [t for t in study.trials if t.user_attrs.get("pruned_reason") == "routing collapse"]
    print(f"trials: {len(states)} total, {states.count(optuna.trial.TrialState.COMPLETE)} completed, "
          f"{states.count(optuna.trial.TrialState.PRUNED)} pruned ({len(collapsed)} for routing collapse)")
    print(f"best trial {study.best_trial.number}: objective {study.best_value:.5f}")
    print(f"best params: {study.best_params}")
    save_config(dict(study.best_params), PROJECT_ROOT / BEST_CONFIG)
    print(f"saved {BEST_CONFIG}")


def run_final(cfg, task, device, warmup_epochs, finetune_epochs, max_batches):
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
    run = init_wandb(task["wandb_group"], dict(params, warmup_epochs=warmup_epochs, finetune_epochs=finetune_epochs),
                     run_name=f"final-{name}", tags=["final"])
    try:
        best = fit(cfg, task, params, warmup_epochs, finetune_epochs, device, run=run,
                   checkpoint_path=checkpoint_path, max_batches=max_batches)
        for key, value in best.items():
            run.summary[f"best/{key}"] = value
        log_checkpoint_artifact(run, checkpoint_path, name, metadata=dict(params, **best))
    finally:
        run.finish()
    print(f"best epoch {best['epoch']}: objective {best['objective']:.5f}  PSNR {best['psnr']:.2f}  "
          f"gate accuracy {best['gate_accuracy']:.3f}")
    print(f"saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Task 3 soft mixture-of-experts.")
    parser.add_argument("--mode", choices=["optuna", "final"], required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--trials", type=int, help="override the number of Optuna trials")
    parser.add_argument("--warmup-epochs", type=int, help="override the number of warm-up epochs")
    parser.add_argument("--finetune-epochs", type=int, help="override the number of fine-tuning epochs")
    parser.add_argument("--num-workers", type=int, help="override num_workers from the base config")
    parser.add_argument("--max-batches", type=int, help="limit batches per epoch (quick tests only)")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    task = load_config(PROJECT_ROOT / TASK_CONFIG)
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    device = get_device()
    print(f"device: {device}")

    stage = task["optuna"] if args.mode == "optuna" else task["final"]
    warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else stage["warmup_epochs"]
    finetune_epochs = args.finetune_epochs if args.finetune_epochs is not None else stage["finetune_epochs"]
    if args.mode == "optuna":
        run_optuna(cfg, task, device, args.trials or task["optuna"]["n_trials"],
                   warmup_epochs, finetune_epochs, args.max_batches)
    else:
        run_final(cfg, task, device, warmup_epochs, finetune_epochs, args.max_batches)


if __name__ == "__main__":
    main()
