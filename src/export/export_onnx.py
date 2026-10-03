"""Export trained models to ONNX and check them against PyTorch.

Run:  python -m src.export.export_onnx --task task1

For every exported model the same inputs are run through PyTorch and
through ONNX Runtime, and the largest absolute difference is reported.
"""

import argparse
import json

import numpy as np
import onnxruntime as ort
import torch

from src.data.manifests import load_manifest
from src.data.pet_dataset import PetManifestDataset, load_images
from src.evaluation.eval_udae import load_autoencoder
from src.utils.config import PROJECT_ROOT, load_config

OPSET = 17
TOLERANCE = 1e-4  # largest allowed difference between PyTorch and ONNX outputs


def export_model(model, example, onnx_path, input_names, output_names):
    """Write `model` to an ONNX file. The batch size is left dynamic."""
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {name: {0: "batch"} for name in list(input_names) + list(output_names)}
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=list(input_names),
        output_names=list(output_names),
        dynamic_axes=dynamic_axes,
        opset_version=OPSET,
        dynamo=False,  # the classic exporter; needs no extra packages
    )


def verify_model(model, inputs, onnx_path, output_names):
    """Run the same inputs through PyTorch and ONNX Runtime.

    Returns {output name: largest absolute difference}.
    """
    with torch.no_grad():
        torch_outputs = model(inputs)
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = (torch_outputs,)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    onnx_outputs = session.run(list(output_names), {input_name: inputs.numpy()})

    differences = {}
    for name, torch_out, onnx_out in zip(output_names, torch_outputs, onnx_outputs):
        differences[name] = float(np.abs(torch_out.numpy() - onnx_out).max())
    return differences


def validation_batch(cfg, num_items=16):
    """Real corrupted validation images to verify on (all four conditions)."""
    images = load_images(PROJECT_ROOT / cfg["data"]["pet_cache_dir"], "val")
    entries = load_manifest(PROJECT_ROOT / cfg["data"]["manifests_dir"] / "pet_val_manifest.jsonl")
    dataset = PetManifestDataset(images, entries)
    return torch.stack([dataset[i][0] for i in range(num_items)])


def export_task1(cfg, checkpoint=None):
    """Task 1: the universal denoising autoencoder."""
    task = load_config(PROJECT_ROOT / "configs/task1_udae.yaml")
    name = task["checkpoint_name"]
    checkpoint_path = checkpoint or PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt"
    onnx_path = PROJECT_ROOT / "models" / f"{name}.onnx"

    # Export and verification both run on the CPU, in eval mode.
    model, _ = load_autoencoder(checkpoint_path, torch.device("cpu"))
    inputs = validation_batch(cfg)
    export_model(model, inputs[:1], onnx_path, ["input"], ["output"])
    differences = verify_model(model, inputs, onnx_path, ["output"])
    return name, onnx_path, differences


def export_task2(cfg, checkpoint=None):
    """Task 2: the classifier and the three specialists, as four separate files.

    Hard routing itself (pick one branch, identity for clean) is done by the
    backend from the classifier's probabilities, so it is not part of a graph.
    """
    from src.evaluation.eval_task2 import load_classifier
    from src.training.train_specialists import SPECIALISTS

    classifier_task = load_config(PROJECT_ROOT / "configs/task2_classifier.yaml")
    specialist_task = load_config(PROJECT_ROOT / "configs/task2_specialists.yaml")
    checkpoints_dir = PROJECT_ROOT / cfg["checkpoints_dir"]
    inputs = validation_batch(cfg)
    exported = []

    name = classifier_task["checkpoint_name"]
    model, _ = load_classifier(checkpoints_dir / f"{name}.pt", torch.device("cpu"))
    onnx_path = PROJECT_ROOT / "models" / f"{name}.onnx"
    export_model(model, inputs[:1], onnx_path, ["input"], ["logits"])
    exported.append((name, onnx_path, verify_model(model, inputs, onnx_path, ["logits"])))

    for specialist in SPECIALISTS:
        name = f"{specialist_task['checkpoint_prefix']}_{specialist}"
        model, _ = load_autoencoder(checkpoints_dir / f"{name}.pt", torch.device("cpu"))
        onnx_path = PROJECT_ROOT / "models" / f"{name}.onnx"
        export_model(model, inputs[:1], onnx_path, ["input"], ["output"])
        exported.append((name, onnx_path, verify_model(model, inputs, onnx_path, ["output"])))
    return exported


def export_task3(cfg, checkpoint=None):
    """Task 3: the complete mixture (gate, three experts, mixing) as ONE graph.

    The temperature is stored inside the model, so it is part of the graph.
    Outputs: the restored image, the four routing weights and the gate logits.
    """
    from src.evaluation.eval_moe import load_moe

    task = load_config(PROJECT_ROOT / "configs/task3_moe.yaml")
    name = task["checkpoint_name"]
    checkpoint_path = checkpoint or PROJECT_ROOT / cfg["checkpoints_dir"] / f"{name}.pt"
    onnx_path = PROJECT_ROOT / "models" / f"{name}.onnx"

    model, _ = load_moe(checkpoint_path, torch.device("cpu"))
    inputs = validation_batch(cfg)
    output_names = ["output", "weights", "logits"]
    export_model(model, inputs[:1], onnx_path, ["input"], output_names)
    return name, onnx_path, verify_model(model, inputs, onnx_path, output_names)


def export_task4(cfg, checkpoint=None):
    """Task 4: the generator only (the discriminator is a training component).

    The exported graph takes a photo in [0, 1] and a style index (0, 1, 2)
    and returns the sketch in [0, 1].
    """
    from src.evaluation.eval_cgan import load_generator
    from src.models.cgan import GeneratorForExport

    task = load_config(PROJECT_ROOT / "configs/task4_cgan.yaml")
    checkpoint_path = checkpoint or PROJECT_ROOT / cfg["checkpoints_dir"] / f"{task['checkpoint_name']}.pt"
    onnx_path = PROJECT_ROOT / "models" / f"{task['generator_name']}.onnx"

    generator, _ = load_generator(checkpoint_path, torch.device("cpu"))
    model = GeneratorForExport(generator).eval()
    # Verify on random photos in all three styles (the FS2K cache may not exist on every machine).
    torch.manual_seed(0)
    photos = torch.rand(6, 3, 128, 128)
    styles = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (photos[:1], styles[:1]), str(onnx_path),
        input_names=["photo", "style"], output_names=["sketch"],
        dynamic_axes={"photo": {0: "batch"}, "style": {0: "batch"}, "sketch": {0: "batch"}},
        opset_version=OPSET, dynamo=False,
    )
    with torch.no_grad():
        expected = model(photos, styles).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["sketch"], {"photo": photos.numpy(), "style": styles.numpy()})[0]
    return task["generator_name"], onnx_path, {"sketch": float(np.abs(expected - actual).max())}


TASKS = {"task1": export_task1, "task2": export_task2, "task3": export_task3, "task4": export_task4}


def main():
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX and verify it.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", help="override the default checkpoint path")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    exported = TASKS[args.task](cfg, args.checkpoint)
    if isinstance(exported, tuple):  # a task that exports a single model
        exported = [exported]

    worst = 0.0
    for name, onnx_path, differences in exported:
        size_mb = onnx_path.stat().st_size / 1e6
        print(f"exported {onnx_path} ({size_mb:.1f} MB, opset {OPSET})")
        for output_name, difference in differences.items():
            status = "OK" if difference <= TOLERANCE else "MISMATCH"
            print(f"  {output_name}: max |PyTorch - ONNX| = {difference:.2e}  [{status}]")
            worst = max(worst, difference)

        # Keep the numbers for the report.
        results_path = PROJECT_ROOT / "report" / "results" / f"{name}_onnx_check.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({"model": name, "opset": OPSET, "tolerance": TOLERANCE,
                       "size_mb": round(size_mb, 2), "max_abs_diff": differences}, f, indent=1)

    if worst > TOLERANCE:
        raise SystemExit("ONNX output does not match PyTorch within the tolerance.")


if __name__ == "__main__":
    main()
