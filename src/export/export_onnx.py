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


TASKS = {"task1": export_task1}


def main():
    parser = argparse.ArgumentParser(description="Export a trained model to ONNX and verify it.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--checkpoint", help="override the default checkpoint path")
    args = parser.parse_args()

    cfg = load_config(PROJECT_ROOT / args.config)
    name, onnx_path, differences = TASKS[args.task](cfg, args.checkpoint)

    size_mb = onnx_path.stat().st_size / 1e6
    print(f"exported {onnx_path} ({size_mb:.1f} MB, opset {OPSET})")
    for output_name, difference in differences.items():
        status = "OK" if difference <= TOLERANCE else "MISMATCH"
        print(f"  {output_name}: max |PyTorch - ONNX| = {difference:.2e}  [{status}]")

    # Keep the numbers for the report.
    results_path = PROJECT_ROOT / "report" / "results" / f"{name}_onnx_check.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({"model": name, "opset": OPSET, "tolerance": TOLERANCE,
                   "size_mb": round(size_mb, 2), "max_abs_diff": differences}, f, indent=1)

    if max(differences.values()) > TOLERANCE:
        raise SystemExit("ONNX output does not match PyTorch within the tolerance.")


if __name__ == "__main__":
    main()
