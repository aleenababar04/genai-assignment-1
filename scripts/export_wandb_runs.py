"""Export a summary of every W&B run of the project to report/results/wandb_runs.csv.

Run from the repository root (needs WANDB_API_KEY in .env or the environment):

    python scripts/export_wandb_runs.py

The W&B project lives in a team workspace that cannot be made public, so this
file keeps an offline copy of the experiment records: one row per run with its
group, name, state, duration, hyperparameters and best validation results.
"""

import csv
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so "import src" works

import wandb  # noqa: E402

from src.utils.tracking import DEFAULT_WANDB_PROJECT  # noqa: E402  (also loads .env)

OUT = Path(__file__).resolve().parents[1] / "report" / "results" / "wandb_runs.csv"


def main():
    api = wandb.Api()
    project = os.environ.get("WANDB_PROJECT", DEFAULT_WANDB_PROJECT)
    path = f"{api.default_entity}/{project}"
    rows = []
    for run in api.runs(path):
        summary = {k: v for k, v in run.summary.items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        rows.append({
            "group": run.group or "",
            "name": run.name,
            "state": run.state,
            "tags": ";".join(run.tags),
            "created_at": run.created_at,
            "runtime_min": round((run.summary.get("_runtime") or 0) / 60, 2),
            "config": json.dumps(config, sort_keys=True),
            "summary": json.dumps(summary, sort_keys=True),
            "url": run.url,
        })
    rows.sort(key=lambda r: (r["group"], r["created_at"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"exported {len(rows)} runs of {path} to {OUT}")


if __name__ == "__main__":
    main()
