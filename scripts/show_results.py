"""Print every key result of the project in one readable summary.

Run from the repository root:

    python scripts/show_results.py

It only READS files (report/results, manifests, optuna_studies, models); it
changes nothing. Use it to see the numbers quoted in the report, or to check
that the files in the repository agree with the report.
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "report" / "results"


def read_csv(name):
    """Rows of a results file as a list of dicts (empty if the file is missing)."""
    path = RESULTS / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def title(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def restoration_table(rows):
    """Print one 'metrics per condition and severity' table (Tasks 1-3)."""
    print(f"{'condition':<12}{'severity':<9}{'images':>7}  {'PSNR in -> out (dB)':<22}{'SSIM in -> out':<18}{'L1':>7}")
    for r in rows:
        psnr_in = "inf" if float(r["input_psnr"]) >= 99.9 else f"{float(r['input_psnr']):.2f}"
        print(f"{r['condition']:<12}{r['severity']:<9}{int(r['count']):>7}  "
              f"{psnr_in + ' -> ' + format(float(r['psnr']), '.2f'):<22}"
              f"{float(r['input_ssim']):.3f} -> {float(r['ssim']):.3f}   {float(r['l1']):.4f}")


def summary_line(label, rows):
    """One line: the 'corrupted / all' row of a restoration table."""
    for r in rows:
        if r["condition"] == "corrupted":
            print(f"  {label:<28} PSNR {float(r['input_psnr']):.2f} -> {float(r['psnr']):.2f} dB   "
                  f"SSIM {float(r['input_ssim']):.3f} -> {float(r['ssim']):.3f}")


def data_section():
    title("DATA SPLITS")
    pet = ROOT / "manifests" / "pet_split.json"
    if pet.exists():
        split = json.load(open(pet, encoding="utf-8"))
        print(f"Oxford-IIIT Pet : train {len(split['train'])}, validation {len(split['val'])}, test {len(split['test'])} images (seed {split['seed']})")
    for name, label in (("pet_val_manifest.jsonl", "validation manifest"), ("pet_test_manifest.jsonl", "test manifest")):
        path = ROOT / "manifests" / name
        if path.exists():
            print(f"Pet {label:<20}: {sum(1 for _ in open(path, encoding='utf-8'))} stored corruptions")
    fs2k = ROOT / "manifests" / "fs2k_split.json"
    if fs2k.exists():
        split = json.load(open(fs2k, encoding="utf-8"))
        for part in ("train", "val", "test"):
            styles = [r["style"] for r in split[part]]
            counts = {f"style {s + 1}": styles.count(s) for s in sorted(set(styles))}
            print(f"FS2K {part:<5}: {len(styles)} photo-sketch pairs {counts}")


def task1_section():
    title("TASK 1 - universal denoising autoencoder (no skip connection)")
    rows = read_csv("task1_udae_test_metrics.csv")
    if not rows:
        return print("(no results file)")
    restoration_table(rows)
    print("\nAblation, one 16x16 skip connection ('corrupted / all' rows):")
    summary_line("no skip (the model we use)", rows)
    summary_line("with skip", read_csv("task1_udae_skip_test_metrics.csv"))


def task2_section():
    title("TASK 2 - classifier + specialists + hard routing")
    per_class = read_csv("task2_classifier_per_class.csv")
    by_sev = read_csv("task2_classifier_by_severity.csv")
    if by_sev:
        total = sum(int(r["count"]) for r in by_sev)
        correct = sum(float(r["accuracy"]) * int(r["count"]) for r in by_sev)
        print(f"Classifier test accuracy: {correct / total:.4f}  ({int(round(total - correct))} wrong out of {total})")
    if per_class:
        print(f"\n{'class':<14}{'precision':>10}{'recall':>9}{'F1':>8}{'images':>9}")
        for r in per_class:
            print(f"{r['class']:<14}{float(r['precision']):>10.4f}{float(r['recall']):>9.4f}{float(r['f1']):>8.4f}{int(r['support']):>9}")
        for key, name in (("precision", "macro precision"), ("recall", "macro recall"), ("f1", "macro F1")):
            print(f"{name:<14}{sum(float(r[key]) for r in per_class) / len(per_class):>10.4f}")
    print("\nClassifier accuracy per condition/severity (and the most common wrong answer):")
    for r in by_sev:
        wrong = f"  most common mistake: {r['most_common_wrong']}" if r["most_common_wrong"] else ""
        print(f"  {r['condition']:<12}{r['severity']:<8}{float(r['accuracy']):.4f}{wrong}")
    print("\nRestoration, 'corrupted / all' rows:")
    summary_line("oracle routing", read_csv("task2_oracle_test_metrics.csv"))
    summary_line("predicted routing", read_csv("task2_predicted_test_metrics.csv"))
    print("\nMisrouted images (true class -> branch it was sent to):")
    for r in read_csv("task2_misrouting.csv"):
        print(f"  {r['true']:<10} -> {r['routed_to']:<11} {int(r['count']):>4} images   SSIM change vs. oracle routing: {-float(r['ssim_loss']):+.3f}")


def task3_section():
    title("TASK 3 - soft mixture-of-experts")
    rows = read_csv("task3_moe_test_metrics.csv")
    if not rows:
        return print("(no results file)")
    restoration_table(rows)
    print("\nAverage routing weights per true corruption (the rows of the heatmap):")
    print(f"{'condition':<12}{'severity':<9}{'identity':>9}{'salt-pep.':>10}{'blur':>8}{'occlusion':>10}   top branch")
    for r in read_csv("task3_moe_mean_weights.csv"):
        print(f"{r['condition']:<12}{r['severity']:<9}{float(r['identity']):>9.3f}{float(r['salt_pepper']):>10.3f}"
              f"{float(r['blur']):>8.3f}{float(r['occlusion']):>10.3f}   {r['top_branch']}")
    print("\nExpert health:")
    for r in read_csv("task3_moe_expert_health.csv"):
        print(f"  {r['branch']:<12} top choice for {100 * float(r['top1_share']):5.1f}% of images, "
              f"mean weight on its own corruption {float(r['mean_weight_on_own']):.3f}, on others {float(r['mean_weight_on_others']):.3f}, "
              f"inactive={r['inactive']}, dominates unrelated inputs={r['dominates_unrelated']}")


def task4_section():
    title("TASK 4 - face-to-sketch conditional GAN (official FS2K test split)")
    for r in read_csv("task4_cgan_test_metrics.csv"):
        print(f"  {r['group']:<8} {int(r['count']):>5} pairs   L1 {float(r['l1']):.4f}   PSNR {float(r['psnr']):.2f} dB   SSIM {float(r['ssim']):.3f}")


def optuna_section():
    title("OPTUNA STUDIES (hyperparameter searches)")
    try:
        import optuna
    except ImportError:
        return print("(optuna is not installed; skipped)")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for db in sorted((ROOT / "optuna_studies").glob("*.db")):
        study = optuna.load_study(study_name=db.stem, storage=f"sqlite:///{db.as_posix()}")
        states = [t.state.name for t in study.trials]
        print(f"\n{db.stem}: {len(states)} trials ({states.count('COMPLETE')} completed, {states.count('PRUNED')} pruned), "
              f"{study.direction.name.lower()}")
        print(f"  best trial #{study.best_trial.number}, value {study.best_value:.4f}")
        for name, value in study.best_params.items():
            print(f"    {name:<16} {value:.5g}" if isinstance(value, float) else f"    {name:<16} {value}")


def onnx_section():
    title("ONNX EXPORT CHECKS (PyTorch output vs. ONNX output; must be below 1e-4)")
    for path in sorted(RESULTS.glob("*_onnx_check.json")):
        d = json.load(open(path, encoding="utf-8"))
        worst = max(d["max_abs_diff"].values())
        verdict = "OK" if worst <= d["tolerance"] else "MISMATCH"
        print(f"  {d['model']:<36} {d['size_mb']:>7.1f} MB   max difference {worst:.2e}   {verdict}")
    models = ROOT / "models"
    present = sorted(p.name for p in models.glob("*.onnx")) if models.exists() else []
    print(f"\nONNX files currently in models/: {len(present)} of 7")


if __name__ == "__main__":
    data_section()
    task1_section()
    task2_section()
    task3_section()
    task4_section()
    optuna_section()
    onnx_section()
