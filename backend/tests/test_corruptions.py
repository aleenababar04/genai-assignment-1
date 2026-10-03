"""The backend corruptions must match the training definitions in src/data/corruptions.py."""

import numpy as np
import pytest

pytest.importorskip("cv2")

from app import corruptions  # noqa: E402


def random_image(seed=0):
    # Values in [0.1, 0.9], so pixels set to exactly 0 or 1 are easy to spot.
    return (0.1 + 0.8 * np.random.default_rng(seed).random((3, 128, 128))).astype(np.float32)


@pytest.mark.parametrize("severity", corruptions.SEVERITIES)
def test_blur_matches_torchvision(severity):
    torch = pytest.importorskip("torch")
    tf = pytest.importorskip("torchvision.transforms.functional")
    image = np.random.default_rng(1).random((3, 128, 128), dtype=np.float32)
    kernel_size, sigma = corruptions.TEST_BLUR[severity]

    expected = tf.gaussian_blur(torch.from_numpy(image), [kernel_size, kernel_size], [sigma, sigma])
    expected = expected.clamp(0.0, 1.0).numpy()
    ours, settings = corruptions.apply(image, "blur", severity, seed=0)

    assert np.abs(ours - expected).max() < 1e-4
    assert settings["kernel_size"] == kernel_size and settings["sigma"] == sigma


def test_fixed_severities_match_training_code():
    source = pytest.importorskip("src.data.corruptions")
    assert corruptions.TEST_SP_PROB == source.TEST_SP_PROB
    assert corruptions.TEST_BLUR == source.TEST_BLUR
    assert corruptions.TEST_OCCLUSION == source.TEST_OCCLUSION


@pytest.mark.parametrize("severity", corruptions.SEVERITIES)
def test_occlusion_rectangles_and_coverage(severity):
    num_rects, target = corruptions.TEST_OCCLUSION[severity]
    image = random_image()
    for seed in range(20):
        out, settings = corruptions.apply(image, "occlusion", severity, seed=seed)
        rects = settings["rectangles"]
        assert len(rects) == num_rects
        assert abs(settings["coverage"] - target) <= corruptions.OCC_COVERAGE_TOLERANCE
        # No two rectangles overlap.
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                assert not corruptions._rects_overlap(rects[i], rects[j])
        # The black area of the image equals the reported coverage.
        black = (out == 0.0).all(axis=0).mean()
        assert abs(black - settings["coverage"]) < 1e-4


@pytest.mark.parametrize("severity", corruptions.SEVERITIES)
def test_salt_and_pepper_fraction(severity):
    prob = corruptions.TEST_SP_PROB[severity]
    image = random_image()
    out, settings = corruptions.apply(image, "salt_pepper", severity, seed=7)
    assert settings["probability"] == prob

    changed = (out != image).any(axis=0)
    white = (out == 1.0).all(axis=0)
    black = (out == 0.0).all(axis=0)
    # Every changed pixel is fully white or fully black (all channels together).
    assert np.array_equal(changed, white | black)
    assert abs(changed.mean() - prob) < 0.015
    # About half salt, half pepper.
    assert abs(white.sum() / changed.sum() - 0.5) < 0.1


def test_seed_makes_corruptions_reproducible():
    image = random_image()
    for name in ["salt_pepper", "occlusion"]:
        first, settings = corruptions.apply(image, name, "high", seed=123)
        second, _ = corruptions.apply(image, name, "high", seed=123)
        other, _ = corruptions.apply(image, name, "high", seed=124)
        assert settings["seed"] == 123
        assert np.array_equal(first, second)
        assert not np.array_equal(first, other)


def test_missing_seed_is_drawn_and_returned():
    image = random_image()
    first, settings = corruptions.apply(image, "salt_pepper", "medium")
    assert isinstance(settings["seed"], int)
    again, _ = corruptions.apply(image, "salt_pepper", "medium", seed=settings["seed"])
    assert np.array_equal(first, again)


def test_none_and_input_not_modified():
    image = random_image()
    copy = image.copy()
    out, settings = corruptions.apply(image, "none", "high", seed=1)
    assert settings == {"type": "none"}
    assert np.array_equal(out, image)
    for name in ["salt_pepper", "blur", "occlusion"]:
        corruptions.apply(image, name, "high", seed=1)
    assert np.array_equal(image, copy)
