"""Plot an Optuna study for the report: optimisation history and parameter importances.

Run:  python -m src.evaluation.plot_optuna --study task1_udae
Writes report/figures/<study>_optuna.png
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna

from src.utils.config import PROJECT_ROOT
from src.utils.tracking import get_study_storage


def main():
    parser = argparse.ArgumentParser(description="Plot an Optuna study.")
    parser.add_argument("--study", required=True, help="study name, e.g. task1_udae")
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=args.study, storage=get_study_storage(args.study))
    trials = [t for t in study.trials if t.value is not None or t.intermediate_values]

    fig, (ax_hist, ax_imp) = plt.subplots(1, 2, figsize=(10, 3.4))

    # Left: objective of every finished trial, plus the best value found so far.
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    ax_hist.scatter([t.number for t in done], [t.value for t in done], s=18, label="completed")
    # A pruned trial has no final value; show its last reported intermediate value.
    ax_hist.scatter([t.number for t in pruned if t.intermediate_values],
                    [list(t.intermediate_values.values())[-1] for t in pruned if t.intermediate_values],
                    s=18, marker="x", label="pruned (last reported)")
    best_so_far, best = [], None
    for t in sorted(done, key=lambda t: t.number):
        better = best is None or (t.value < best if study.direction.name == "MINIMIZE" else t.value > best)
        best = t.value if better else best
        best_so_far.append((t.number, best))
    ax_hist.step([n for n, _ in best_so_far], [v for _, v in best_so_far], where="post",
                 color="black", linewidth=1, label="best so far")
    ax_hist.set_xlabel("trial")
    ax_hist.set_ylabel("validation objective")
    ax_hist.set_title(f"{args.study}: optimisation history", fontsize=9)
    ax_hist.legend(fontsize=7)

    # Right: which hyperparameters mattered most (fANOVA importance over completed trials).
    importances = optuna.importance.get_param_importances(study)
    names = list(importances)[::-1]
    ax_imp.barh(names, [importances[n] for n in names])
    ax_imp.set_xlabel("importance")
    ax_imp.set_title("hyperparameter importance", fontsize=9)

    fig.tight_layout()
    out_path = PROJECT_ROOT / "report" / "figures" / f"{args.study}_optuna.png"
    fig.savefig(out_path, dpi=150)
    print(f"{len(trials)} trials plotted; importances: "
          + ", ".join(f"{k} {v:.2f}" for k, v in importances.items()))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
