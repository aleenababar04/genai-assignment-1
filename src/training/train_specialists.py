"""Task 2b: train the three specialist autoencoders.

  python -m src.training.train_specialists --mode optuna   # one shared search for all three
  python -m src.training.train_specialists --mode final    # train each specialist independently

Each specialist only ever sees its own corruption: the salt-and-pepper
specialist is trained on salt-and-pepper images only, and so on. The
architecture is the Task 1 ConvAutoencoder, without dropout.
"""

import argparse

import optuna
import torch
from torch.utils.data import DataLoader

from src.data import corruptions as C
from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, PetTrainDataset, load_images
from src.models.autoencoder import count_parameters
from src.training.train_udae import build_model, log_samples, train_one_epoch, validate
from src.utils.config import PROJECT_ROOT, get_device, load_config, save_config
from src.utils.seed import make_generator, seed_worker, set_seed
from src.utils.tracking import create_study, init_wandb, log_checkpoint_artifact

TASK_CONFIG = "configs/task2_specialists.yaml"
BEST_CONFIG = "configs/task2_specialists_best.yaml"

# Specialist name -> the corruption it is responsible for.
SPECIALISTS = {"salt_pepper": C.SALT_PEPPER, "blur": C.BLUR, "occlusion": C.OCCLUSION}


def make_specialist_data(cfg, condition, batch_size, device):
    """Training loader with ONE corruption type, and the matching validation items."""
    cache_dir = PROJECT_ROOT / cfg["data"]["pet_cache_dir"]
    train_set = PetTrainDataset(load_images(cache_dir, "train"), conditions=(condition,))
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
    # Only the validation entries of this corruption (736 of the 2,944).
    entries = load_manifest(PROJECT_ROOT / cfg["data"]["manifests_dir"] / "pet_val_manifest.jsonl")
    entries = [e for e in entries if e["label"] == condition]
    val_set = PetManifestDataset(load_images(cache_dir, "val"), entries)
    return train_loader, val_set


def save_specialist(path, model, params, epoch, val_stats):
    """Same checkpoint format as Task 1, so the same loader works for both."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_args": {
                "base_channels": params["base_channels"],
                "latent_channels": params["latent_channels"],
                "dropout": params["dropout"],
                "skip": False,
            },
            "params": params,
            "epoch": epoch,
            "val": val_stats,
        },
        path,
    )


def fit_specialists(cfg, task, params, epochs, device, names, runs=None, trial=None,
                    checkpoint_dir=None, max_batches=None):
    """Train the named specialists side by side, one epoch each in turn.

    Each specialist has its own model, optimiser, data and best checkpoint;
    they only share the epoch loop, so that Optuna can be told the mean
    validation objective after every epoch (needed for pruning).
    `runs` maps a specialist name to its W&B run (or is None).
    Returns {name: validation stats of that specialist's best epoch}.
    """
    set_seed(cfg["seed"])
    params = dict(params, dropout=0.0, skip=False)
    state = {}
    for name in names:
        train_loader, val_set = make_specialist_data(cfg, SPECIALISTS[name], params["batch_size"], device)
        model = build_model(params).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        sample_indices = list(range(min(task["final"]["num_samples"], len(val_set))))
        state[name] = {"loader": train_loader, "val_set": val_set, "model": model,
                       "optimizer": optimizer, "scheduler": scheduler,
                       "samples": sample_indices, "best": None}

    for epoch in range(1, epochs + 1):
        objectives = []
        for name in names:
            s = state[name]
            train_stats = train_one_epoch(s["model"], s["loader"], s["optimizer"], params["alpha"], device, max_batches)
            val_stats = validate(s["model"], s["val_set"], device, task["objective_alpha"])
            s["scheduler"].step()
            objectives.append(val_stats["objective"])

            print(f"epoch {epoch:3d}/{epochs}  {name:<12} train loss {train_stats['loss']:.4f}  "
                  f"val objective {val_stats['objective']:.4f}  PSNR {val_stats['psnr']:.2f}  "
                  f"SSIM {val_stats['ssim']:.4f}")
            run = runs.get(name) if runs else None
            if run is not None:
                log = {"epoch": epoch, "lr": s["optimizer"].param_groups[0]["lr"]}
                log.update({f"train/{k}": v for k, v in train_stats.items()})
                log.update({f"val/{k}": v for k, v in val_stats.items()})
                run.log(log)
                if trial is None and epoch % task["final"]["sample_every"] == 0:
                    log_samples(run, s["model"], s["val_set"], s["samples"], device, epoch)

            if s["best"] is None or val_stats["objective"] < s["best"]["objective"]:
                s["best"] = dict(val_stats, epoch=epoch)
                if checkpoint_dir is not None:
                    path = checkpoint_dir / f"{task['checkpoint_prefix']}_{name}.pt"
                    save_specialist(path, s["model"], params, epoch, val_stats)

        if trial is not None:
            mean_objective = sum(objectives) / len(objectives)
            trial.report(mean_objective, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return {name: state[name]["best"] for name in names}


def suggest_params(trial, space):
    """Draw one hyperparameter set from the search space in the config."""
    return {
        "lr": trial.suggest_float("lr", space["lr"][0], space["lr"][1], log=True),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "latent_channels": trial.suggest_categorical("latent_channels", space["latent_channels"]),
        "base_channels": trial.suggest_categorical("base_channels", space["base_channels"]),
        "alpha": trial.suggest_float("alpha", space["alpha"][0], space["alpha"][1]),
    }


def run_optuna(cfg, task, device, n_trials, epochs, max_batches):
    """One shared search: a trial's score is the mean objective of all three specialists."""
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=task["optuna"]["pruner_startup_trials"],
        n_warmup_steps=task["optuna"]["pruner_warmup_epochs"],
    )
    study = create_study(task["study_name"], direction="minimize", pruner=pruner)
    names = list(SPECIALISTS)

    def objective(trial):
        params = suggest_params(trial, task["search_space"])
        run = init_wandb(task["wandb_group"], dict(params, epochs=epochs),
                         run_name=f"trial-{trial.number:03d}", tags=["optuna-trial"])
        try:
            # One W&B run per trial; the specialists' metrics are logged with a prefix.
            runs = {name: _PrefixedRun(run, name) for name in names}
            best = fit_specialists(cfg, task, params, epochs, device, names, runs=runs,
                                   trial=trial, max_batches=max_batches)
        finally:
            run.finish()
        for name in names:
            trial.set_user_attr(f"{name}_objective", best[name]["objective"])
            trial.set_user_attr(f"{name}_psnr", best[name]["psnr"])
            trial.set_user_attr(f"{name}_ssim", best[name]["ssim"])
        return sum(best[name]["objective"] for name in names) / len(names)

    study.optimize(objective, n_trials=n_trials)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"trials: {len(study.trials)} total, {len(completed)} completed, {len(pruned)} pruned")
    print(f"best trial {study.best_trial.number}: mean objective {study.best_value:.5f}")
    print(f"best params: {study.best_params}")
    save_config(dict(study.best_params), PROJECT_ROOT / BEST_CONFIG)
    print(f"saved {BEST_CONFIG}")


class _PrefixedRun:
    """Lets three specialists share one W&B run by prefixing their metric names."""

    def __init__(self, run, prefix):
        self.run = run
        self.prefix = prefix

    def log(self, data):
        self.run.log({(k if k == "epoch" else f"{self.prefix}/{k}"): v for k, v in data.items()})


def run_final(cfg, task, device, epochs, names, max_batches):
    """Train each specialist on its own, with its own W&B run and checkpoint."""
    best_path = PROJECT_ROOT / BEST_CONFIG
    if best_path.exists():
        params = load_config(best_path)
        print(f"using Optuna result from {BEST_CONFIG}")
    else:
        params = dict(task["defaults"])
        print("no Optuna result found, using the defaults from the task config")
    print(f"parameters per specialist: {count_parameters(build_model(dict(params, dropout=0.0))):,}")

    checkpoint_dir = PROJECT_ROOT / cfg["checkpoints_dir"]
    for name in names:
        checkpoint_name = f"{task['checkpoint_prefix']}_{name}"
        run = init_wandb(task["wandb_group"], dict(params, epochs=epochs, specialist=name),
                         run_name=f"final-{checkpoint_name}", tags=["final", name])
        try:
            best = fit_specialists(cfg, task, params, epochs, device, [name], runs={name: run},
                                   checkpoint_dir=checkpoint_dir, max_batches=max_batches)[name]
            for key, value in best.items():
                run.summary[f"best/{key}"] = value
            log_checkpoint_artifact(run, checkpoint_dir / f"{checkpoint_name}.pt", checkpoint_name,
                                    metadata=dict(params, **best))
        finally:
            run.finish()
        print(f"{name}: best epoch {best['epoch']}  PSNR {best['psnr']:.2f}  SSIM {best['ssim']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train the Task 2 specialist autoencoders.")
    parser.add_argument("--mode", choices=["optuna", "final"], required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--trials", type=int, help="override the number of Optuna trials")
    parser.add_argument("--epochs", type=int, help="override the number of epochs")
    parser.add_argument("--only", choices=list(SPECIALISTS), help="final mode: train just one specialist")
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
        names = [args.only] if args.only else list(SPECIALISTS)
        run_final(cfg, task, device, epochs=args.epochs or task["final"]["epochs"],
                  names=names, max_batches=args.max_batches)


if __name__ == "__main__":
    main()
