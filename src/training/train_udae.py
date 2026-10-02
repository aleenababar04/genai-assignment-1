"""Task 1: train the universal denoising autoencoder.

Two modes:
  python -m src.training.train_udae --mode optuna   # hyperparameter search
  python -m src.training.train_udae --mode final    # full training with the best settings

The model never sees the corruption label: it gets a corrupted image and
must reproduce the clean one, whatever the corruption was.
"""

import argparse

import optuna
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, PetTrainDataset, load_images
from src.evaluation.metrics import evaluate_restoration, mean_metrics
from src.models.autoencoder import ConvAutoencoder, count_parameters
from src.training.losses import restoration_loss
from src.utils.config import PROJECT_ROOT, get_device, load_config, save_config
from src.utils.seed import make_generator, seed_worker, set_seed
from src.utils.tracking import create_study, init_wandb, log_checkpoint_artifact

TASK_CONFIG = "configs/task1_udae.yaml"
BEST_CONFIG = "configs/task1_udae_best.yaml"


def build_model(params):
    """Create the autoencoder described by a hyperparameter dict."""
    return ConvAutoencoder(
        base_channels=params["base_channels"],
        latent_channels=params["latent_channels"],
        dropout=params["dropout"],
        skip=params.get("skip", False),
    )


def make_loaders(cfg, batch_size, device):
    """Training loader (random corruptions) and validation set (fixed manifest)."""
    cache_dir = PROJECT_ROOT / cfg["data"]["pet_cache_dir"]
    manifests_dir = PROJECT_ROOT / cfg["data"]["manifests_dir"]

    train_set = PetTrainDataset(load_images(cache_dir, "train"))
    val_set = PetManifestDataset(
        load_images(cache_dir, "val"),
        load_manifest(manifests_dir / "pet_val_manifest.jsonl"),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg["num_workers"],
        worker_init_fn=seed_worker,
        generator=make_generator(cfg["seed"]),
        pin_memory=(device.type == "cuda"),
    )
    return train_loader, val_set


def train_one_epoch(model, loader, optimizer, alpha, device, max_batches=None):
    """One pass over the training data. Returns the mean loss, L1 and SSIM."""
    model.train()
    totals = {"loss": 0.0, "l1": 0.0, "ssim": 0.0}
    num_batches = 0
    for corrupted, clean, _ in loader:  # the label is not used in Task 1
        corrupted, clean = corrupted.to(device), clean.to(device)

        restored = model(corrupted)
        loss, l1, ssim_value = restoration_loss(restored, clean, alpha)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        totals["loss"] += loss.item()
        totals["l1"] += l1.item()
        totals["ssim"] += ssim_value.item()
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break
    return {name: total / num_batches for name, total in totals.items()}


def validate(model, val_set, device, objective_alpha):
    """Score the model on the fixed validation manifest."""
    model.eval()
    metrics = evaluate_restoration(model, val_set, device)
    stats = mean_metrics(metrics)
    # Fixed weighting, independent of the alpha used for training.
    stats["objective"] = objective_alpha * stats["l1"] + (1 - objective_alpha) * (1 - stats["ssim"])
    return stats


def log_samples(run, model, val_set, indices, device, epoch):
    """Log a grid to W&B: corrupted inputs (top), restorations (middle), clean targets (bottom)."""
    model.eval()
    corrupted = torch.stack([val_set[i][0] for i in indices])
    clean = torch.stack([val_set[i][1] for i in indices])
    with torch.no_grad():
        restored = model(corrupted.to(device)).cpu()
    grid = make_grid(torch.cat([corrupted, restored, clean]), nrow=len(indices), padding=2)
    run.log({"samples": wandb.Image(grid, caption="corrupted / restored / clean"), "epoch": epoch})


def fit(cfg, task, params, epochs, device, run=None, trial=None, checkpoint_path=None, max_batches=None):
    """Train one model and return the validation stats of its best epoch."""
    set_seed(cfg["seed"])
    train_loader, val_set = make_loaders(cfg, params["batch_size"], device)

    model = build_model(params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    # Cosine schedule: the learning rate falls smoothly towards zero.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # The same validation items are logged every time, so progress is comparable.
    # Entries 1, 2, 3 of each image are its salt-and-pepper, blur and occlusion versions.
    sample_indices = [i for i in range(len(val_set)) if val_set.entries[i]["label"] != 0]
    sample_indices = sample_indices[: task["final"]["num_samples"]]

    best = None
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(model, train_loader, optimizer, params["alpha"], device, max_batches)
        val_stats = validate(model, val_set, device, task["objective_alpha"])
        scheduler.step()

        print(
            f"epoch {epoch:3d}/{epochs}  train loss {train_stats['loss']:.4f}  "
            f"val objective {val_stats['objective']:.4f}  "
            f"PSNR {val_stats['psnr']:.2f}  SSIM {val_stats['ssim']:.4f}"
        )
        if run is not None:
            log = {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            log.update({f"train/{k}": v for k, v in train_stats.items()})
            log.update({f"val/{k}": v for k, v in val_stats.items()})
            run.log(log)
            if trial is None and epoch % task["final"]["sample_every"] == 0:
                log_samples(run, model, val_set, sample_indices, device, epoch)

        if best is None or val_stats["objective"] < best["objective"]:
            best = dict(val_stats, epoch=epoch)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "model_args": {
                            "base_channels": params["base_channels"],
                            "latent_channels": params["latent_channels"],
                            "dropout": params["dropout"],
                            "skip": params.get("skip", False),
                        },
                        "params": params,
                        "epoch": epoch,
                        "val": val_stats,
                    },
                    checkpoint_path,
                )

        if trial is not None:
            # Tell Optuna how the trial is doing so it can stop weak ones early.
            trial.report(val_stats["objective"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best


def suggest_params(trial, space):
    """Draw one hyperparameter set from the search space in the config."""
    return {
        "lr": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "latent_channels": trial.suggest_categorical("latent_channels", space["latent_channels"]),
        "base_channels": trial.suggest_categorical("base_channels", space["base_channels"]),
        "dropout": trial.suggest_float("dropout", space["dropout"][0], space["dropout"][1]),
        "alpha": trial.suggest_float("alpha", space["alpha"][0], space["alpha"][1]),
    }


def run_optuna(cfg, task, device, n_trials, epochs, max_batches):
    """Hyperparameter search. The best settings are written to BEST_CONFIG."""
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=task["optuna"]["pruner_startup_trials"],
        n_warmup_steps=task["optuna"]["pruner_warmup_epochs"],
    )
    study = create_study(task["study_name"], direction="minimize", pruner=pruner)

    def objective(trial):
        params = suggest_params(trial, task["search_space"])
        run = init_wandb(task["wandb_group"], dict(params, epochs=epochs),
                         run_name=f"trial-{trial.number:03d}", tags=["optuna-trial"])
        try:
            best = fit(cfg, task, params, epochs, device, run=run, trial=trial, max_batches=max_batches)
        finally:
            run.finish()
        # Keep the plain metrics with the trial, for the report tables.
        for name in ("l1", "psnr", "ssim", "epoch"):
            trial.set_user_attr(name, best[name])
        return best["objective"]

    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"trials: {len(study.trials)} total, {len(completed)} completed, {len(pruned)} pruned")
    print(f"best trial {study.best_trial.number}: objective {study.best_value:.5f}")
    print(f"best params: {study.best_params}")

    best_params = dict(study.best_params, skip=False)
    save_config(best_params, PROJECT_ROOT / BEST_CONFIG)
    print(f"saved {BEST_CONFIG}")


def run_final(cfg, task, device, epochs, skip, max_batches):
    """Full training with the best settings found by Optuna (or the defaults)."""
    best_path = PROJECT_ROOT / BEST_CONFIG
    if best_path.exists():
        params = load_config(best_path)
        print(f"using Optuna result from {BEST_CONFIG}")
    else:
        params = dict(task["defaults"])
        print("no Optuna result found, using the defaults from the task config")
    params["skip"] = skip

    # The skip-connection ablation is saved under its own name.
    name = task["checkpoint_name"] + ("_skip" if skip else "")
    checkpoint_path = PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt"
    print(f"parameters: {count_parameters(build_model(params)):,}")

    run = init_wandb(task["wandb_group"], dict(params, epochs=epochs), run_name=f"final-{name}", tags=["final"])
    try:
        best = fit(cfg, task, params, epochs, device, run=run,
                   checkpoint_path=checkpoint_path, max_batches=max_batches)
        for key, value in best.items():
            run.summary[f"best/{key}"] = value
        log_checkpoint_artifact(run, checkpoint_path, name, metadata=dict(params, **best))
    finally:
        run.finish()
    print(f"best epoch {best['epoch']}: objective {best['objective']:.5f}  "
          f"PSNR {best['psnr']:.2f}  SSIM {best['ssim']:.4f}")
    print(f"saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Task 1 universal denoising autoencoder.")
    parser.add_argument("--mode", choices=["optuna", "final"], required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--trials", type=int, help="override the number of Optuna trials")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--num-workers", type=int, help="override num_workers from the base config")
    parser.add_argument("--skip", action="store_true", help="final mode: train the limited-skip ablation")
    parser.add_argument("--max-batches", type=int, help="limit batches per epoch (quick tests only)")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    task = load_config(PROJECT_ROOT / TASK_CONFIG)
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    device = get_device()
    print(f"device: {device}")

    if args.mode == "optuna":
        run_optuna(
            cfg, task, device,
            n_trials=args.trials or task["optuna"]["n_trials"],
            epochs=args.epochs or task["optuna"]["epochs_per_trial"],
            max_batches=args.max_batches,
        )
    else:
        run_final(cfg, task, device, epochs=args.epochs or task["final"]["epochs"],
                  skip=args.skip, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
