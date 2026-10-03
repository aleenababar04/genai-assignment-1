"""Task 4: train the style-conditioned face-to-sketch GAN.

  python -m src.training.train_cgan --mode optuna   # hyperparameter search (shorter runs)
  python -m src.training.train_cgan --mode final    # full training with the best settings

Each step first updates the discriminator (real pairs -> "real", generated
pairs -> "fake"), then the generator (fool the discriminator and stay close
to the true sketch in L1).
"""

import argparse

import numpy as np
import optuna
import torch
import wandb
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from src.data.fs2k_dataset import FS2KDataset, to_unit_range
from src.evaluation.metrics import l1_per_image, psnr_per_image, ssim_per_image
from src.models.autoencoder import count_parameters
from src.models.cgan import PatchDiscriminator, StyleUNetGenerator, discriminator_loss, generator_loss
from src.utils.config import PROJECT_ROOT, get_device, load_config, save_config
from src.utils.seed import make_generator, seed_worker, set_seed
from src.utils.tracking import create_study, init_wandb, log_checkpoint_artifact

TASK_CONFIG = "configs/task4_cgan.yaml"
BEST_CONFIG = "configs/task4_cgan_best.yaml"


def fs2k_cache_dir(cfg):
    return PROJECT_ROOT / cfg["data"].get("fs2k_cache_dir", "data/fs2k_128")


def build_networks(params, out_channels):
    """Generator and discriminator described by a hyperparameter dict."""
    generator_args = {
        "base_channels": params["base_channels"],
        "style_dim": params["style_dim"],
        "dropout": params["dropout"],
        "out_channels": out_channels,
    }
    discriminator_args = {
        "base_channels": params["base_channels"],
        "style_dim": params["style_dim"],
        "in_sketch_channels": out_channels,
    }
    return (StyleUNetGenerator(**generator_args), PatchDiscriminator(**discriminator_args),
            generator_args, discriminator_args)


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, l1_weight, device, max_batches=None):
    """One pass over the training pairs. Returns the four losses the brief asks for."""
    generator.train()
    discriminator.train()
    totals = {"d_real": 0.0, "d_fake": 0.0, "g_adv": 0.0, "g_l1": 0.0}
    num_batches = 0
    for photo, sketch, style in loader:
        photo, sketch, style = photo.to(device), sketch.to(device), style.to(device)
        fake = generator(photo, style)

        # 1. Discriminator: real pair -> 1, generated pair -> 0.
        #    detach() stops this loss from changing the generator.
        real_logits = discriminator(photo, sketch, style)
        fake_logits = discriminator(photo, fake.detach(), style)
        d_loss, d_real, d_fake = discriminator_loss(real_logits, fake_logits)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # 2. Generator: make the discriminator say "real", and match the true sketch.
        fake_logits = discriminator(photo, fake, style)
        g_loss, g_adv, g_l1 = generator_loss(fake_logits, fake, sketch, l1_weight)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        totals["d_real"] += d_real.item()
        totals["d_fake"] += d_fake.item()
        totals["g_adv"] += g_adv.item()
        totals["g_l1"] += g_l1.item()
        num_batches += 1
        if max_batches is not None and num_batches >= max_batches:
            break
    return {name: total / num_batches for name, total in totals.items()}


@torch.no_grad()
def generate_all(generator, dataset, device, batch_size=32):
    """Generated sketches for a whole split, in [0, 1], with the true sketches and styles."""
    generator.eval()  # dropout off: the same photo and style always give the same sketch
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    fakes, reals, styles = [], [], []
    for photo, sketch, style in loader:
        fake = generator(photo.to(device), style.to(device)).cpu()
        fakes.append(to_unit_range(fake))
        reals.append(to_unit_range(sketch))
        styles.append(style)
    return torch.cat(fakes), torch.cat(reals), torch.cat(styles)


def sketch_metrics(fakes, reals, styles):
    """Per-image L1, PSNR and SSIM (in [0, 1]), as NumPy arrays, plus the styles."""
    return {
        "l1": l1_per_image(fakes, reals).numpy(),
        "psnr": psnr_per_image(fakes, reals).numpy(),
        "ssim": ssim_per_image(fakes, reals).numpy(),
        "style": styles.numpy(),
    }


def validate(generator, val_set, device, objective_l1_weight):
    """Mean L1/PSNR/SSIM on the validation split, overall and per style."""
    metrics = sketch_metrics(*generate_all(generator, val_set, device))
    stats = {name: float(metrics[name].mean()) for name in ("l1", "psnr", "ssim")}
    for s in range(3):
        mask = metrics["style"] == s
        if mask.any():
            stats[f"ssim_style{s + 1}"] = float(metrics["ssim"][mask].mean())
    stats["objective"] = objective_l1_weight * stats["l1"] + (1 - objective_l1_weight) * (1 - stats["ssim"])
    return stats


def sample_indices(val_set, per_style):
    """The same validation photos every time: the first `per_style` of each style."""
    chosen = []
    for s in range(3):
        chosen += [i for i in range(len(val_set)) if int(val_set.styles[i]) == s][:per_style]
    return chosen


@torch.no_grad()
def log_samples(run, generator, val_set, indices, device, epoch):
    """Grid: photo, true sketch, generated sketch, then the same photo in all three styles."""
    generator.eval()
    photos = torch.stack([val_set[i][0] for i in indices]).to(device)
    sketches = torch.stack([val_set[i][1] for i in indices])
    styles = torch.tensor([int(val_set[i][2]) for i in indices], device=device)
    columns = [to_unit_range(photos.cpu()), to_unit_range(sketches), to_unit_range(generator(photos, styles).cpu())]
    for s in range(3):
        fixed = torch.full_like(styles, s)
        columns.append(to_unit_range(generator(photos, fixed).cpu()))
    # One row per photo: photo | true | generated | style 1 | style 2 | style 3
    rows = torch.stack(columns, dim=1).flatten(0, 1)
    if rows.shape[1] == 1:
        rows = rows.repeat(1, 3, 1, 1)
    grid = make_grid(rows, nrow=len(columns), padding=2)
    run.log({"samples": wandb.Image(grid, caption="photo | true sketch | generated | style 1 | style 2 | style 3"),
             "epoch": epoch})


def fit(cfg, task, params, epochs, device, run=None, trial=None, checkpoint_path=None, max_batches=None):
    """Train one GAN and return the validation stats of its best epoch."""
    set_seed(cfg["seed"])
    cache_dir = fs2k_cache_dir(cfg)
    train_set = FS2KDataset(cache_dir, "train", augment=True)
    val_set = FS2KDataset(cache_dir, "val")
    loader = DataLoader(train_set, batch_size=params["batch_size"], shuffle=True, drop_last=True,
                        num_workers=cfg["num_workers"], worker_init_fn=seed_worker,
                        generator=make_generator(cfg["seed"]), pin_memory=(device.type == "cuda"))

    generator, discriminator, generator_args, discriminator_args = build_networks(params, task["out_channels"])
    generator, discriminator = generator.to(device), discriminator.to(device)
    betas = tuple(task["adam_betas"])
    opt_g = torch.optim.Adam(generator.parameters(), lr=params["lr_g"], betas=betas)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=params["lr_d"], betas=betas)
    indices = sample_indices(val_set, task["final"]["samples_per_style"])

    best = None
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(generator, discriminator, loader, opt_g, opt_d,
                                      params["l1_weight"], device, max_batches)
        val_stats = validate(generator, val_set, device, task["objective_l1_weight"])

        print(f"epoch {epoch:3d}/{epochs}  D real {train_stats['d_real']:.3f}  D fake {train_stats['d_fake']:.3f}  "
              f"G adv {train_stats['g_adv']:.3f}  G L1 {train_stats['g_l1']:.4f}  "
              f"val SSIM {val_stats['ssim']:.4f}  val objective {val_stats['objective']:.4f}")
        if run is not None:
            log = {"epoch": epoch}
            log.update({f"train/{k}": v for k, v in train_stats.items()})
            log.update({f"val/{k}": v for k, v in val_stats.items()})
            run.log(log)
            if trial is None and (epoch == 1 or epoch % task["final"]["sample_every"] == 0):
                log_samples(run, generator, val_set, indices, device, epoch)

        if best is None or val_stats["objective"] < best["objective"]:
            best = dict(val_stats, epoch=epoch)
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"generator_state": generator.state_dict(),
                            "discriminator_state": discriminator.state_dict(),
                            "generator_args": generator_args, "discriminator_args": discriminator_args,
                            "params": params, "epoch": epoch, "val": val_stats}, checkpoint_path)

        if trial is not None:
            trial.report(val_stats["objective"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best


def suggest_params(trial, space):
    """Draw one hyperparameter set from the search space in the config."""
    return {
        "lr_g": trial.suggest_float("lr_g", space["lr_g"][0], space["lr_g"][1], log=True),
        "lr_d": trial.suggest_float("lr_d", space["lr_d"][0], space["lr_d"][1], log=True),
        "batch_size": trial.suggest_categorical("batch_size", space["batch_size"]),
        "base_channels": trial.suggest_categorical("base_channels", space["base_channels"]),
        "dropout": trial.suggest_float("dropout", space["dropout"][0], space["dropout"][1]),
        "style_dim": trial.suggest_categorical("style_dim", space["style_dim"]),
        "l1_weight": trial.suggest_float("l1_weight", space["l1_weight"][0], space["l1_weight"][1], log=True),
    }


def run_optuna(cfg, task, device, n_trials, epochs, max_batches):
    """Hyperparameter search with shorter runs; the winner is retrained in final mode."""
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=task["optuna"]["pruner_startup_trials"],
        n_warmup_steps=task["optuna"]["pruner_warmup_epochs"],
    )
    study = create_study(task["study_name"], direction="minimize", pruner=pruner)
    if len(study.trials) == 0:
        study.enqueue_trial(dict(task["defaults"]))  # trial 0 = the pix2pix settings

    def objective(trial):
        params = suggest_params(trial, task["search_space"])
        run = init_wandb(task["wandb_group"], dict(params, epochs=epochs),
                         run_name=f"trial-{trial.number:03d}", tags=["optuna-trial"])
        try:
            best = fit(cfg, task, params, epochs, device, run=run, trial=trial, max_batches=max_batches)
        finally:
            run.finish()
        for name in ("l1", "psnr", "ssim", "epoch"):
            trial.set_user_attr(name, best[name])
        return best["objective"]

    study.optimize(objective, n_trials=n_trials)
    states = [t.state for t in study.trials]
    print(f"trials: {len(states)} total, {states.count(optuna.trial.TrialState.COMPLETE)} completed, "
          f"{states.count(optuna.trial.TrialState.PRUNED)} pruned")
    print(f"best trial {study.best_trial.number}: objective {study.best_value:.5f}")
    print(f"best params: {study.best_params}")
    save_config(dict(study.best_params), PROJECT_ROOT / BEST_CONFIG)
    print(f"saved {BEST_CONFIG}")


def run_final(cfg, task, device, epochs, max_batches):
    """Full training schedule with the best settings found by Optuna (or the defaults)."""
    best_path = PROJECT_ROOT / BEST_CONFIG
    if best_path.exists():
        params = load_config(best_path)
        print(f"using Optuna result from {BEST_CONFIG}")
    else:
        params = dict(task["defaults"])
        print("no Optuna result found, using the defaults from the task config")
    generator, discriminator, _, _ = build_networks(params, task["out_channels"])
    print(f"parameters: generator {count_parameters(generator):,}  discriminator {count_parameters(discriminator):,}")

    name = task["checkpoint_name"]
    checkpoint_path = PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt"
    run = init_wandb(task["wandb_group"], dict(params, epochs=epochs), run_name=f"final-{name}", tags=["final"])
    try:
        best = fit(cfg, task, params, epochs, device, run=run, checkpoint_path=checkpoint_path, max_batches=max_batches)
        for key, value in best.items():
            run.summary[f"best/{key}"] = value
        log_checkpoint_artifact(run, checkpoint_path, name, metadata=dict(params, **best))
    finally:
        run.finish()
    print(f"best epoch {best['epoch']}: objective {best['objective']:.5f}  SSIM {best['ssim']:.4f}")
    print(f"saved {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the Task 4 face-to-sketch conditional GAN.")
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
        run_optuna(cfg, task, device, args.trials or task["optuna"]["n_trials"],
                   args.epochs or task["optuna"]["epochs_per_trial"], args.max_batches)
    else:
        run_final(cfg, task, device, args.epochs or task["final"]["epochs"], args.max_batches)


if __name__ == "__main__":
    main()
