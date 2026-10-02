"""Experiment tracking helpers: Weights & Biases runs and Optuna studies."""

import os
from pathlib import Path

import optuna
import wandb
from dotenv import load_dotenv

from src.utils.config import PROJECT_ROOT

# Load secrets/settings (e.g. WANDB_API_KEY, WANDB_MODE) from <repo>/.env if present.
ENV_FILE = PROJECT_ROOT / ".env"
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)

DEFAULT_WANDB_PROJECT = "genai-a1"

# Folder holding one SQLite file per Optuna study.
STUDIES_DIR = PROJECT_ROOT / "optuna_studies"


def init_wandb(
    task: str,
    config: dict,
    run_name: str | None = None,
    tags: list[str] | None = None,
):
    """Start a W&B run in the shared project, grouped by task.

    The project name comes from the WANDB_PROJECT environment variable
    (default "genai-a1"). W&B reads WANDB_MODE by itself, so setting
    WANDB_MODE=disabled or WANDB_MODE=offline works without code changes.
    """
    project = os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    run = wandb.init(
        project=project,
        group=task,
        name=run_name,
        tags=tags,
        config=config,
    )
    return run


def get_study_storage(study_name: str) -> str:
    """Return the SQLite URL for a study: sqlite:///<optuna_studies>/<study_name>.db"""
    studies_dir = Path(STUDIES_DIR).resolve()
    studies_dir.mkdir(parents=True, exist_ok=True)
    db_path = studies_dir / f"{study_name}.db"
    # as_posix() gives forward slashes, which the URL needs on Windows too.
    return "sqlite:///" + db_path.as_posix()


def create_study(study_name: str, direction: str = "minimize", pruner=None):
    """Create an Optuna study stored in SQLite, or reload it if it already exists."""
    study = optuna.create_study(
        study_name=study_name,
        direction=direction,
        storage=get_study_storage(study_name),
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=pruner,
        load_if_exists=True,
    )
    return study


def log_checkpoint_artifact(run, path, name: str, metadata: dict | None = None):
    """Upload a checkpoint file to W&B as an artifact of type "model"."""
    artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
    artifact.add_file(str(path))
    run.log_artifact(artifact)
    return artifact
