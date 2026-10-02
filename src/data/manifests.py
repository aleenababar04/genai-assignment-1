"""Fixed corruption manifests for the validation and test sets.

Training corruptions are sampled fresh every time an image is loaded.
Validation and test corruptions must be the same on every run, so they are
generated once here and saved to disk. Each manifest entry stores everything
needed to rebuild one corrupted image: the corruption type, the severity,
the blur settings, the rectangle coordinates and the random seed.

Run once:  python -m src.data.manifests
"""

import argparse
import json
from pathlib import Path

import torch

from src.data import corruptions as C
from src.utils.config import PROJECT_ROOT, load_config

# Seeds of test entries start here so they never coincide with validation seeds.
TEST_SEED_OFFSET = 1_000_000


def make_entry(entry_id, image_index, image_name, condition, severity, params, seed):
    """One manifest row. `image_index` is the row in the split's .npy cache."""
    return {
        "id": entry_id,
        "image_index": image_index,
        "image": image_name,
        "condition": C.CONDITION_NAMES[condition],
        "label": condition,
        "severity": severity,
        "params": params,
        "seed": seed,
    }


def build_val_manifest(names, image_size=128, seed=42):
    """Four entries per validation image: one for each input condition.

    Severities are drawn from the training ranges, using a generator seeded
    with the entry's own seed, so the manifest is identical on every run.
    """
    entries = []
    for image_index, name in enumerate(names):
        for condition in (C.CLEAN, C.SALT_PEPPER, C.BLUR, C.OCCLUSION):
            entry_seed = seed + len(entries)
            generator = torch.Generator().manual_seed(entry_seed)
            params = C.sample_params(condition, image_size, image_size, generator=generator)
            severity = "none" if condition == C.CLEAN else "sampled"
            entries.append(
                make_entry(len(entries), image_index, name, condition, severity, params, entry_seed)
            )
    return entries


def build_test_manifest(names, image_size=128, seed=42):
    """Ten entries per test image: clean, plus 3 corruptions x 3 fixed severities."""
    entries = []
    for image_index, name in enumerate(names):
        # (condition, severity, params) for this image; occlusion is filled in below
        plan = [(C.CLEAN, "none", {})]
        for severity in C.SEVERITIES:
            plan.append((C.SALT_PEPPER, severity, {"prob": C.TEST_SP_PROB[severity]}))
        for severity in C.SEVERITIES:
            kernel_size, sigma = C.TEST_BLUR[severity]
            plan.append((C.BLUR, severity, {"kernel_size": kernel_size, "sigma": sigma}))
        for severity in C.SEVERITIES:
            plan.append((C.OCCLUSION, severity, None))

        for condition, severity, params in plan:
            entry_seed = seed + TEST_SEED_OFFSET + len(entries)
            if condition == C.OCCLUSION:
                num_rects, coverage = C.TEST_OCCLUSION[severity]
                generator = torch.Generator().manual_seed(entry_seed)
                params = C.sample_occlusion_params(
                    image_size, image_size, generator=generator,
                    num_rects=num_rects, coverage=coverage,
                )
            entries.append(
                make_entry(len(entries), image_index, name, condition, severity, params, entry_seed)
            )
    return entries


def save_manifest(entries, path):
    """Write one JSON object per line (JSON Lines)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def load_manifest(path):
    """Read a manifest written by `save_manifest`."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Generate the validation and test corruption manifests.")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--force", action="store_true", help="overwrite existing manifests")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    manifests_dir = PROJECT_ROOT / cfg["data"]["manifests_dir"]
    with open(manifests_dir / "pet_split.json", "r", encoding="utf-8") as f:
        split = json.load(f)

    jobs = [
        ("pet_val_manifest.jsonl", build_val_manifest, split["val"]),
        ("pet_test_manifest.jsonl", build_test_manifest, split["test"]),
    ]
    for file_name, build, names in jobs:
        path = manifests_dir / file_name
        if path.exists() and not args.force:
            # The manifests are meant to be generated once and then kept fixed.
            print(f"{path.name} already exists, keeping it (use --force to regenerate)")
            continue
        entries = build(names, image_size=cfg["image_size"], seed=cfg["seed"])
        save_manifest(entries, path)
        print(f"wrote {len(entries)} entries for {len(names)} images to {path.name}")


if __name__ == "__main__":
    main()
