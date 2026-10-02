"""Tests for src/data/corruptions.py."""

import pytest
import torch

from src.data import corruptions as C

SIZE = 128


def make_generator(seed: int) -> torch.Generator:
    """Return a seeded generator so every test is reproducible."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def random_image(seed: int = 0) -> torch.Tensor:
    """Return a random (3, SIZE, SIZE) image with values in [0, 1]."""
    return torch.rand(3, SIZE, SIZE, generator=make_generator(seed))


def rects_overlap(a, b) -> bool:
    """Independent overlap check: paint both rectangles and look for shared pixels."""
    mask_a = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    mask_b = torch.zeros(SIZE, SIZE, dtype=torch.bool)
    mask_a[a[0]:a[0] + a[2], a[1]:a[1] + a[3]] = True
    mask_b[b[0]:b[0] + b[2], b[1]:b[1] + b[3]] = True
    return bool((mask_a & mask_b).any())


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

def test_constants_match_the_brief():
    assert (C.CLEAN, C.SALT_PEPPER, C.BLUR, C.OCCLUSION) == (0, 1, 2, 3)
    assert C.CONDITION_NAMES == ["clean", "salt_pepper", "blur", "occlusion"]
    assert C.SP_PROB_RANGE == (0.02, 0.15)
    assert C.BLUR_KERNEL_SIZES == (3, 5, 7)
    assert C.BLUR_SIGMA_RANGE == (0.5, 2.5)
    assert C.OCC_NUM_RECTS == (1, 3)
    assert C.OCC_COVERAGE_RANGE == (0.10, 0.35)
    assert C.SEVERITIES == ["low", "medium", "high"]
    assert C.TEST_SP_PROB == {"low": 0.03, "medium": 0.08, "high": 0.15}
    assert C.TEST_BLUR == {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}
    assert C.TEST_OCCLUSION == {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}


# ---------------------------------------------------------------------------
# Properties shared by every condition
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("condition", [C.CLEAN, C.SALT_PEPPER, C.BLUR, C.OCCLUSION])
def test_input_not_modified_and_output_valid(condition):
    img = random_image()
    original = img.clone()
    generator = make_generator(1)

    params = C.sample_params(condition, SIZE, SIZE, generator)
    out = C.apply_corruption(img, condition, params, generator)

    assert torch.equal(img, original)  # input untouched
    assert out is not img  # a new tensor
    assert out.shape == img.shape
    assert out.dtype == img.dtype
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_clean_returns_equal_tensor():
    img = random_image()
    out = C.apply_corruption(img, C.CLEAN, {})
    assert torch.equal(out, img)
    assert C.sample_params(C.CLEAN, SIZE, SIZE) == {}


def test_unknown_condition_raises():
    with pytest.raises(ValueError):
        C.sample_params(99, SIZE, SIZE)
    with pytest.raises(ValueError):
        C.apply_corruption(random_image(), 99, {})


# ---------------------------------------------------------------------------
# Salt-and-pepper
# ---------------------------------------------------------------------------

def test_salt_and_pepper_fraction_and_values():
    img = torch.full((3, SIZE, SIZE), 0.5)  # mid-grey: every hit is visible
    prob = 0.10
    out = C.salt_and_pepper(img, prob, make_generator(0))

    # A pixel counts as changed if any of its channels changed.
    changed = (out != img).any(dim=0)
    fraction = changed.float().mean().item()
    assert abs(fraction - prob) < 0.02

    # Changed pixels are pure white or pure black in all three channels.
    white = (out == 1.0).all(dim=0)
    black = (out == 0.0).all(dim=0)
    assert torch.equal(changed, white | black)

    # About half salt, half pepper.
    salt_share = white.sum().item() / changed.sum().item()
    assert abs(salt_share - 0.5) < 0.05

    # Untouched pixels keep their value.
    assert torch.all(out[:, ~changed] == 0.5)


def test_salt_and_pepper_same_seed_same_output():
    img = random_image()
    out_a = C.salt_and_pepper(img, 0.1, make_generator(7))
    out_b = C.salt_and_pepper(img, 0.1, make_generator(7))
    out_c = C.salt_and_pepper(img, 0.1, make_generator(8))
    assert torch.equal(out_a, out_b)
    assert not torch.equal(out_a, out_c)


def test_salt_and_pepper_prob_zero_changes_nothing():
    img = random_image()
    assert torch.equal(C.salt_and_pepper(img, 0.0, make_generator(0)), img)


def test_sampled_salt_pepper_params_in_range():
    generator = make_generator(0)
    for _ in range(300):
        params = C.sample_salt_pepper_params(generator)
        assert isinstance(params["prob"], float)
        assert C.SP_PROB_RANGE[0] <= params["prob"] <= C.SP_PROB_RANGE[1]


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------

def test_blur_leaves_constant_image_unchanged():
    img = torch.full((3, SIZE, SIZE), 0.3)
    out = C.gaussian_blur(img, 7, 2.5)
    assert torch.allclose(out, img, atol=1e-5)


def test_blur_reduces_variance_of_noise():
    img = random_image()
    out = C.gaussian_blur(img, 5, 1.5)
    assert out.var() < img.var()


def test_sampled_blur_params_in_range():
    generator = make_generator(0)
    seen_kernels = set()
    for _ in range(300):
        params = C.sample_blur_params(generator)
        assert isinstance(params["kernel_size"], int)
        assert isinstance(params["sigma"], float)
        assert params["kernel_size"] in C.BLUR_KERNEL_SIZES
        assert C.BLUR_SIGMA_RANGE[0] <= params["sigma"] <= C.BLUR_SIGMA_RANGE[1]
        seen_kernels.add(params["kernel_size"])
    assert seen_kernels == set(C.BLUR_KERNEL_SIZES)


# ---------------------------------------------------------------------------
# Occlusion
# ---------------------------------------------------------------------------

def test_occlude_blacks_out_only_the_rectangle():
    img = torch.full((3, 10, 10), 0.5)
    out = C.occlude(img, [[2, 3, 4, 5]])  # rows 2..5, columns 3..7
    assert torch.all(out[:, 2:6, 3:8] == 0.0)
    assert (out == 0.0).sum().item() == 3 * 4 * 5
    assert (out == 0.5).sum().item() == 3 * (100 - 20)


def test_coverage_of_counts_overlap_once():
    # Two 4x4 squares overlapping in a 2x2 block: 16 + 16 - 4 = 28 pixels.
    rects = [[0, 0, 4, 4], [2, 2, 4, 4]]
    assert C.coverage_of(rects, 10, 10) == pytest.approx(0.28)
    assert C.coverage_of([], 10, 10) == 0.0
    assert C.coverage_of([[0, 0, 10, 10]], 10, 10) == 1.0


def test_sampled_occlusion_is_valid():
    generator = make_generator(0)
    seen_counts = set()
    for _ in range(300):
        params = C.sample_occlusion_params(SIZE, SIZE, generator)
        rects = params["rects"]
        seen_counts.add(len(rects))

        assert 1 <= len(rects) <= 3
        for top, left, rect_h, rect_w in rects:
            assert all(isinstance(v, int) for v in (top, left, rect_h, rect_w))
            assert rect_h >= 1 and rect_w >= 1
            assert top >= 0 and left >= 0
            assert top + rect_h <= SIZE and left + rect_w <= SIZE

        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                assert not rects_overlap(rects[i], rects[j])

        assert isinstance(params["coverage"], float)
        assert 0.10 <= params["coverage"] <= 0.35
        assert params["coverage"] == pytest.approx(C.coverage_of(rects, SIZE, SIZE))
        # No overlap, so the union coverage equals the sum of the areas.
        total_area = sum(rect_h * rect_w for _, _, rect_h, rect_w in rects)
        assert params["coverage"] == pytest.approx(total_area / (SIZE * SIZE))
    assert seen_counts == {1, 2, 3}


@pytest.mark.parametrize("severity", C.SEVERITIES)
def test_fixed_test_severities(severity):
    num_rects, coverage = C.TEST_OCCLUSION[severity]
    for seed in range(100):
        params = C.sample_occlusion_params(
            SIZE, SIZE, make_generator(seed), num_rects=num_rects, coverage=coverage
        )
        assert len(params["rects"]) == num_rects
        assert abs(params["coverage"] - coverage) <= 0.01
        assert params["coverage"] == pytest.approx(C.coverage_of(params["rects"], SIZE, SIZE))


def test_occlusion_same_seed_same_params():
    params_a = C.sample_occlusion_params(SIZE, SIZE, make_generator(3))
    params_b = C.sample_occlusion_params(SIZE, SIZE, make_generator(3))
    assert params_a == params_b


def test_occluded_fraction_matches_reported_coverage():
    img = torch.ones(3, SIZE, SIZE)  # white, so black pixels are the occlusion
    params = C.sample_occlusion_params(SIZE, SIZE, make_generator(5))
    out = C.occlude(img, params["rects"])
    black_fraction = (out[0] == 0.0).float().mean().item()
    assert black_fraction == pytest.approx(params["coverage"])


# ---------------------------------------------------------------------------
# corrupt_random
# ---------------------------------------------------------------------------

def test_corrupt_random_gives_all_labels_about_equally():
    img = torch.rand(3, 32, 32, generator=make_generator(0))
    generator = make_generator(0)
    num_draws = 2000
    counts = [0, 0, 0, 0]
    for _ in range(num_draws):
        out, label, params = C.corrupt_random(img, generator=generator)
        assert isinstance(label, int)
        assert isinstance(params, dict)
        assert out.shape == img.shape
        counts[label] += 1
    for count in counts:
        assert abs(count / num_draws - 0.25) < 0.05


def test_corrupt_random_respects_conditions():
    img = random_image()
    generator = make_generator(0)
    for _ in range(50):
        _, label, params = C.corrupt_random(img, conditions=(2,), generator=generator)
        assert label == 2
        assert set(params) == {"kernel_size", "sigma"}


def test_corrupt_random_same_seed_same_result():
    img = random_image()
    out_a, label_a, params_a = C.corrupt_random(img, generator=make_generator(11))
    out_b, label_b, params_b = C.corrupt_random(img, generator=make_generator(11))
    assert label_a == label_b
    assert params_a == params_b
    assert torch.equal(out_a, out_b)
