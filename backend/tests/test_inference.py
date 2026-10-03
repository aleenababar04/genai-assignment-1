"""Tests of the model registry, the inference functions and the image helpers (no web server)."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from conftest import MOE_LOGITS, make_png_bytes

pytest.importorskip("onnxruntime")

from app import imaging, inference  # noqa: E402
from app.models import EXPECTED_MODELS, ModelRegistry, ModelUnavailable  # noqa: E402


def random_image(seed=0):
    return np.random.default_rng(seed).random((3, 128, 128), dtype=np.float32)


def test_registry_loads_all_models(models_dir):
    registry = ModelRegistry(models_dir)
    status = registry.status()
    assert list(status) == EXPECTED_MODELS
    assert all(entry["loaded"] and entry["error"] is None for entry in status.values())


def test_registry_reports_missing_and_broken(tmp_path, models_dir):
    (tmp_path / "task1_udae.onnx").write_bytes((models_dir / "task1_udae.onnx").read_bytes())
    (tmp_path / "task3_moe.onnx").write_bytes(b"this is not an onnx file")
    registry = ModelRegistry(tmp_path)
    status = registry.status()
    assert status["task1_udae.onnx"] == {"loaded": True, "error": None}
    assert status["task2_classifier.onnx"]["loaded"] is False
    assert "not found" in status["task2_classifier.onnx"]["error"]
    assert status["task3_moe.onnx"]["loaded"] is False
    assert "Failed to load" in status["task3_moe.onnx"]["error"]
    with pytest.raises(ModelUnavailable, match="Model task3_moe.onnx is not loaded."):
        registry.get("task3_moe.onnx")


def test_universal(models_dir):
    image = random_image()
    result = inference.universal(ModelRegistry(models_dir), image)
    assert result["output"].shape == (3, 128, 128)
    np.testing.assert_allclose(result["output"], image, atol=1e-6)  # the dummy is the identity
    assert result["inference_ms"] >= 0


def test_hard_routing_clean_uses_identity_bypass(models_dir_for):
    image = random_image(1)
    result = inference.hard_routed(ModelRegistry(models_dir_for("clean")), image)
    assert result["predicted"] == "clean"
    assert result["selected_expert"] == "identity"
    assert result["identity_bypass"] is True
    np.testing.assert_array_equal(result["output"], image)
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6


@pytest.mark.parametrize("forced, expected", [
    ("salt_pepper", lambda x: 0.5 * x),
    ("blur", lambda x: 1.0 - x),
    ("occlusion", lambda x: 0.5 * x + 0.5),
])
def test_hard_routing_runs_the_chosen_specialist(models_dir_for, forced, expected):
    image = random_image(2)
    result = inference.hard_routed(ModelRegistry(models_dir_for(forced)), image)
    assert result["predicted"] == forced
    assert result["selected_expert"] == forced
    assert result["identity_bypass"] is False
    np.testing.assert_allclose(result["output"], expected(image), atol=1e-6)
    assert max(result["probabilities"], key=result["probabilities"].get) == forced


def test_hard_routing_does_not_need_specialists_for_clean(tmp_path, models_dir_for):
    # Only the classifier exists: a clean prediction must still work.
    source = models_dir_for("clean") / "task2_classifier.onnx"
    (tmp_path / "task2_classifier.onnx").write_bytes(source.read_bytes())
    result = inference.hard_routed(ModelRegistry(tmp_path), random_image())
    assert result["identity_bypass"] is True


def test_soft_moe(models_dir):
    result = inference.soft_moe(ModelRegistry(models_dir), random_image())
    weights = result["weights"]
    assert list(weights) == ["identity", "salt_pepper", "blur", "occlusion"]
    assert abs(sum(weights.values()) - 1.0) < 1e-5
    assert result["top_branch"] == ["identity", "salt_pepper", "blur", "occlusion"][int(np.argmax(MOE_LOGITS))]


@pytest.mark.parametrize("style", [1, 2, 3])
def test_sketch(models_dir, style):
    image = random_image()
    result = inference.sketch(ModelRegistry(models_dir), image, style)
    assert result["style"] == style
    assert result["output"].shape == (1, 128, 128)
    np.testing.assert_allclose(result["output"][0], image.mean(axis=0), atol=1e-6)


def test_sketch_rejects_bad_style(models_dir):
    with pytest.raises(ValueError):
        inference.sketch(ModelRegistry(models_dir), random_image(), 4)


def test_decode_upload_matches_training_preprocessing(tmp_path):
    prepare_pet = pytest.importorskip("src.data.prepare_pet")
    data = make_png_bytes(5)
    path = tmp_path / "image.png"
    path.write_bytes(data)
    expected = prepare_pet.load_image(path).transpose(2, 0, 1).astype(np.float32) / 255.0
    decoded = imaging.decode_upload(data)
    assert decoded.shape == (3, 128, 128) and decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)


def test_decode_upload_rejects_bad_data():
    with pytest.raises(imaging.ImageError):
        imaging.decode_upload(b"definitely not an image")
    gif = io.BytesIO()
    Image.new("RGB", (8, 8)).save(gif, format="GIF")
    with pytest.raises(imaging.ImageError, match="Unsupported file type"):
        imaging.decode_upload(gif.getvalue())


def test_png_base64_round_trip():
    image = random_image(3)
    url = imaging.to_png_base64(image)
    assert url.startswith("data:image/png;base64,")
    decoded = np.asarray(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))))
    assert decoded.shape == (128, 128, 3)
    assert np.abs(decoded.astype(np.float32) / 255.0 - image.transpose(1, 2, 0)).max() <= 0.5 / 255 + 1e-6
    grey = imaging.to_png_base64(image[:1])  # 1 channel becomes RGB
    grey_pixels = np.asarray(Image.open(io.BytesIO(base64.b64decode(grey.split(",", 1)[1]))))
    assert grey_pixels.shape == (128, 128, 3)
