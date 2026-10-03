"""Shared test fixtures: tiny dummy ONNX models and sample images in temporary folders.

Run from the repo root:  .venv\\Scripts\\python.exe -m pytest backend/tests -q

The dummy models are built with PyTorch (available in the training venv, not
in the Docker image). They have the exact input/output names and shapes of
the real exported models, but trivial, known behaviour:

  task1_udae                    output = x
  task2_classifier              logits = a constant vector (forces one class)
  task2_specialist_salt_pepper  output = 0.5 * x
  task2_specialist_blur         output = 1 - x
  task2_specialist_occlusion    output = 0.5 * x + 0.5
  task3_moe                     output = x, logits = constant, weights = softmax(logits)
  task4_generator               sketch = grey version of the photo, 1 channel
"""

import io
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make "import app" work when pytest runs from the repo root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# Logits that force each classifier class (clean, salt_pepper, blur, occlusion).
CLASS_INDEX = {"clean": 0, "salt_pepper": 1, "blur": 2, "occlusion": 3}
MOE_LOGITS = [0.5, 2.0, 1.0, -1.0]  # the soft-mixture gate picks salt_pepper most
DEFAULT_CLASS = "blur"  # what the classifier in the default models folder predicts


# ---------------------------------------------------------------------------
# Building dummy ONNX models
# ---------------------------------------------------------------------------

def _build_modules():
    """Define the dummy torch modules (imported lazily so torch is only needed here)."""
    import torch
    import torch.nn as nn

    class Affine(nn.Module):
        """output = scale * x + shift."""

        def __init__(self, scale, shift):
            super().__init__()
            self.scale, self.shift = scale, shift

        def forward(self, x):
            return self.scale * x + self.shift

    class ConstantLogits(nn.Module):
        """Returns fixed logits; "0 * mean(x)" keeps the input in the graph."""

        def __init__(self, logits):
            super().__init__()
            self.register_buffer("logits", torch.tensor([logits], dtype=torch.float32))

        def forward(self, x):
            return self.logits + 0.0 * x.mean(dim=(1, 2, 3), keepdim=False)[:, None]

    class DummyMoE(nn.Module):
        """Identity output with constant gate logits and their softmax weights."""

        def __init__(self, logits):
            super().__init__()
            self.gate = ConstantLogits(logits)

        def forward(self, x):
            logits = self.gate(x)
            return x * 1.0, torch.softmax(logits, dim=1), logits

    class DummyGenerator(nn.Module):
        """1-channel sketch = channel mean of the photo; style is used but has no effect."""

        def forward(self, photo, style):
            grey = photo.mean(dim=1, keepdim=True)
            return grey + 0.0 * style.float().view(-1, 1, 1, 1)

    return torch, Affine, ConstantLogits, DummyMoE, DummyGenerator


def _export(torch, model, example, path, input_names, output_names):
    """Same settings as src/export/export_onnx.py (opset 17, dynamic batch)."""
    dynamic_axes = {name: {0: "batch"} for name in list(input_names) + list(output_names)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # the classic exporter is "legacy"
        torch.onnx.export(model.eval(), example, str(path), input_names=input_names,
                          output_names=output_names, dynamic_axes=dynamic_axes,
                          opset_version=17, dynamo=False)


def build_classifier(path: Path, forced_class: str) -> None:
    """Write a classifier ONNX file that always predicts `forced_class`."""
    torch, _, ConstantLogits, _, _ = _build_modules()
    logits = [0.0, 0.0, 0.0, 0.0]
    logits[CLASS_INDEX[forced_class]] = 5.0
    x = torch.rand(1, 3, 128, 128)
    _export(torch, ConstantLogits(logits), x, path, ["input"], ["logits"])


def build_all_models(models_dir: Path) -> None:
    """Write all seven dummy ONNX files into `models_dir`."""
    torch, Affine, _, DummyMoE, DummyGenerator = _build_modules()
    models_dir.mkdir(parents=True, exist_ok=True)
    x = torch.rand(1, 3, 128, 128)

    _export(torch, Affine(1.0, 0.0), x, models_dir / "task1_udae.onnx", ["input"], ["output"])
    build_classifier(models_dir / "task2_classifier.onnx", DEFAULT_CLASS)
    _export(torch, Affine(0.5, 0.0), x, models_dir / "task2_specialist_salt_pepper.onnx",
            ["input"], ["output"])
    _export(torch, Affine(-1.0, 1.0), x, models_dir / "task2_specialist_blur.onnx", ["input"], ["output"])
    _export(torch, Affine(0.5, 0.5), x, models_dir / "task2_specialist_occlusion.onnx",
            ["input"], ["output"])
    _export(torch, DummyMoE(MOE_LOGITS), x, models_dir / "task3_moe.onnx",
            ["input"], ["output", "weights", "logits"])
    style = torch.zeros(1, dtype=torch.int64)
    _export(torch, DummyGenerator(), (x, style), models_dir / "task4_generator.onnx",
            ["photo", "style"], ["sketch"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def models_dir(tmp_path_factory) -> Path:
    """All seven dummy models; the classifier always predicts DEFAULT_CLASS."""
    pytest.importorskip("torch")
    pytest.importorskip("onnx")
    directory = tmp_path_factory.mktemp("models")
    build_all_models(directory)
    return directory


@pytest.fixture(scope="session")
def models_dir_for(tmp_path_factory, models_dir):
    """Factory: a copy of the dummy models whose classifier predicts a chosen class."""
    cache = {}

    def make(forced_class: str) -> Path:
        if forced_class not in cache:
            directory = tmp_path_factory.mktemp(f"models_{forced_class}")
            for path in models_dir.iterdir():
                shutil.copy(path, directory / path.name)
            build_classifier(directory / "task2_classifier.onnx", forced_class)
            cache[forced_class] = directory
        return cache[forced_class]

    return make


@pytest.fixture(scope="session")
def models_dir_without_moe(tmp_path_factory, models_dir) -> Path:
    """A copy of the dummy models with task3_moe.onnx missing."""
    directory = tmp_path_factory.mktemp("models_no_moe")
    for path in models_dir.iterdir():
        if path.name != "task3_moe.onnx":
            shutil.copy(path, directory / path.name)
    return directory


def make_png_bytes(seed: int = 0, size=(160, 120)) -> bytes:
    """A random RGB PNG (not 128x128, so resizing is exercised)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(20, 236, size=(size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="session")
def samples_dir(tmp_path_factory) -> Path:
    """One pet PNG and one face JPEG, plus a secret file outside the sample folders."""
    directory = tmp_path_factory.mktemp("samples")
    (directory / "pets").mkdir()
    (directory / "faces").mkdir()
    (directory / "pets" / "cat.png").write_bytes(make_png_bytes(1))
    Image.open(io.BytesIO(make_png_bytes(2))).save(directory / "faces" / "person.jpg", format="JPEG")
    (directory / "pets" / "notes.txt").write_text("not an image")  # must not be listed
    (directory / "secret.png").write_bytes(make_png_bytes(3))  # outside pets/faces
    return directory


@pytest.fixture
def make_client(monkeypatch, samples_dir):
    """Factory: a TestClient whose app loaded its models from the given folder."""
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")  # needed by fastapi.testclient
    pytest.importorskip("multipart")  # needed for form uploads
    pytest.importorskip("cv2")
    from fastapi.testclient import TestClient

    from app import config
    from app.main import app

    clients = []

    def make(directory: Path):
        # The app reads these settings at startup / per request, so patching them is enough.
        monkeypatch.setattr(config, "MODELS_DIR", directory)
        monkeypatch.setattr(config, "SAMPLES_DIR", samples_dir)
        client = TestClient(app)
        client.__enter__()  # runs the lifespan, which loads the models
        clients.append(client)
        return client

    yield make
    for client in clients:
        client.__exit__(None, None, None)


@pytest.fixture
def client(make_client, models_dir):
    """A client with all seven dummy models loaded."""
    return make_client(models_dir)
