"""Image corruptions: salt-and-pepper noise, Gaussian blur and occlusion.

Images are float tensors of shape (3, H, W) with values in [0, 1].
Every function returns a NEW tensor and never changes its input.

All randomness comes from torch (never `random` or NumPy), so that the
automatic per-worker seeding of the DataLoader gives every worker different
corruptions. Each random function takes an optional `generator`; None means
"use the global torch random number generator".
"""

import torch
import torchvision.transforms.functional as TF

# Condition labels.
CLEAN, SALT_PEPPER, BLUR, OCCLUSION = 0, 1, 2, 3
CONDITION_NAMES = ["clean", "salt_pepper", "blur", "occlusion"]

# Training ranges (from the assignment brief).
SP_PROB_RANGE = (0.02, 0.15)
BLUR_KERNEL_SIZES = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_NUM_RECTS = (1, 3)  # inclusive
OCC_COVERAGE_RANGE = (0.10, 0.35)

# Fixed test severities (from the assignment brief).
SEVERITIES = ["low", "medium", "high"]
TEST_SP_PROB = {"low": 0.03, "medium": 0.08, "high": 0.15}
TEST_BLUR = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}  # (kernel_size, sigma)
TEST_OCCLUSION = {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}  # (num_rects, coverage)

# Settings of the occlusion sampler.
OCC_SHARE_RANGE = (0.5, 1.5)  # relative area of each rectangle before normalising
OCC_ASPECT_RANGE = (0.5, 2.0)  # height / width of a rectangle
OCC_COVERAGE_TOLERANCE = 0.01  # allowed error for a fixed test coverage
OCC_PLACEMENT_TRIES = 50  # tries to place one rectangle without overlap
OCC_MAX_ATTEMPTS = 1000  # tries to build one complete set of rectangles


# ---------------------------------------------------------------------------
# Small random helpers
# ---------------------------------------------------------------------------

def _uniform(low: float, high: float, generator: torch.Generator | None = None) -> float:
    """Return one Python float drawn uniformly from [low, high)."""
    u = torch.rand(1, generator=generator).item()
    return low + (high - low) * u


def _randint(low: int, high: int, generator: torch.Generator | None = None) -> int:
    """Return one Python int drawn uniformly from low..high (both inclusive)."""
    return int(torch.randint(low, high + 1, (1,), generator=generator).item())


# ---------------------------------------------------------------------------
# Apply functions (deterministic given their arguments)
# ---------------------------------------------------------------------------

def salt_and_pepper(img: torch.Tensor, prob: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """Turn each pixel white or black with probability `prob`.

    A "pixel" is a spatial location, so all 3 channels change together.
    A selected pixel becomes white (salt) or black (pepper) with equal chance.
    """
    _, height, width = img.shape
    # One random number per pixel decides whether the pixel is corrupted.
    selected = torch.rand(height, width, generator=generator) < prob
    # A second random number per pixel decides salt (True) or pepper (False).
    is_salt = torch.rand(height, width, generator=generator) < 0.5

    salt_mask = (selected & is_salt).to(img.device)
    pepper_mask = (selected & ~is_salt).to(img.device)

    out = img.clone()
    out[:, salt_mask] = 1.0  # the mask indexes (H, W); ":" covers all channels
    out[:, pepper_mask] = 0.0
    return out


def gaussian_blur(img: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Blur the image with a square Gaussian kernel."""
    blurred = TF.gaussian_blur(img, [kernel_size, kernel_size], [sigma, sigma])
    # Floating-point rounding can leave values a hair outside [0, 1].
    return blurred.clamp(0.0, 1.0)


def occlude(img: torch.Tensor, rects: list) -> torch.Tensor:
    """Paint black rectangles. `rects` is a list of [top, left, height, width]."""
    out = img.clone()
    for top, left, rect_h, rect_w in rects:
        out[:, top:top + rect_h, left:left + rect_w] = 0.0
    return out


def coverage_of(rects: list, height: int, width: int) -> float:
    """Fraction of the image covered by the union of the rectangles.

    A boolean mask is used, so overlapping areas are counted only once.
    """
    mask = torch.zeros(height, width, dtype=torch.bool)
    for top, left, rect_h, rect_w in rects:
        mask[top:top + rect_h, left:left + rect_w] = True
    return mask.sum().item() / (height * width)


# ---------------------------------------------------------------------------
# Parameter samplers (return plain Python types so they can be saved as JSON)
# ---------------------------------------------------------------------------

def sample_salt_pepper_params(generator: torch.Generator | None = None) -> dict:
    """Draw the noise probability uniformly from SP_PROB_RANGE."""
    prob = _uniform(SP_PROB_RANGE[0], SP_PROB_RANGE[1], generator)
    return {"prob": prob}


def sample_blur_params(generator: torch.Generator | None = None) -> dict:
    """Draw a kernel size from BLUR_KERNEL_SIZES and a sigma from BLUR_SIGMA_RANGE."""
    index = _randint(0, len(BLUR_KERNEL_SIZES) - 1, generator)
    kernel_size = BLUR_KERNEL_SIZES[index]
    sigma = _uniform(BLUR_SIGMA_RANGE[0], BLUR_SIGMA_RANGE[1], generator)
    return {"kernel_size": kernel_size, "sigma": sigma}


def _rects_overlap(a: list, b: list) -> bool:
    """Return True if rectangles a and b ([top, left, height, width]) share any pixel."""
    a_top, a_left, a_h, a_w = a
    b_top, b_left, b_h, b_w = b
    # They are separate if one lies fully above/below or left/right of the other.
    separate_vertically = a_top + a_h <= b_top or b_top + b_h <= a_top
    separate_horizontally = a_left + a_w <= b_left or b_left + b_w <= a_left
    return not (separate_vertically or separate_horizontally)


def _try_build_rects(height: int, width: int, num_rects: int, coverage: float,
                     generator: torch.Generator | None = None):
    """Make one attempt at `num_rects` non-overlapping rectangles.

    Together they should cover about `coverage` of the image.
    Returns the list of rectangles, or None if a rectangle could not be placed.
    """
    # 1. Split the target area between the rectangles with random shares.
    shares = [_uniform(OCC_SHARE_RANGE[0], OCC_SHARE_RANGE[1], generator) for _ in range(num_rects)]
    total_share = sum(shares)
    target_area = coverage * height * width

    rects = []
    for share in shares:
        area = target_area * share / total_share

        # 2. Pick a shape. With aspect = h / w and area = h * w we get
        #    h = sqrt(area * aspect) and w = sqrt(area / aspect).
        aspect = _uniform(OCC_ASPECT_RANGE[0], OCC_ASPECT_RANGE[1], generator)
        rect_h = round((area * aspect) ** 0.5)
        rect_w = round((area / aspect) ** 0.5)
        # Keep the rectangle at least 1 pixel and at most the image size.
        rect_h = max(1, min(rect_h, height))
        rect_w = max(1, min(rect_w, width))

        # 3. Try random positions until the rectangle does not hit the others.
        placed = False
        for _ in range(OCC_PLACEMENT_TRIES):
            top = _randint(0, height - rect_h, generator)
            left = _randint(0, width - rect_w, generator)
            candidate = [top, left, rect_h, rect_w]
            if not any(_rects_overlap(candidate, other) for other in rects):
                rects.append(candidate)
                placed = True
                break
        if not placed:
            return None
    return rects


def sample_occlusion_params(height: int, width: int, generator: torch.Generator | None = None,
                            num_rects: int | None = None, coverage: float | None = None) -> dict:
    """Sample non-overlapping black rectangles for an image of the given size.

    If `num_rects` or `coverage` is None it is drawn from the training range.
    Passing both gives a fixed test severity. The returned "coverage" is the
    real measured coverage, which differs slightly from the target because
    rectangle sides are whole pixels.
    """
    coverage_is_fixed = coverage is not None

    for _ in range(OCC_MAX_ATTEMPTS):
        # Draw whatever the caller did not fix (new values on every attempt).
        n = num_rects if num_rects is not None else _randint(OCC_NUM_RECTS[0], OCC_NUM_RECTS[1], generator)
        if coverage_is_fixed:
            target = coverage
        else:
            target = _uniform(OCC_COVERAGE_RANGE[0], OCC_COVERAGE_RANGE[1], generator)

        rects = _try_build_rects(height, width, n, target, generator)
        if rects is None:
            continue  # a rectangle could not be placed: start again

        real = coverage_of(rects, height, width)
        if coverage_is_fixed:
            accepted = abs(real - target) <= OCC_COVERAGE_TOLERANCE
        else:
            accepted = OCC_COVERAGE_RANGE[0] <= real <= OCC_COVERAGE_RANGE[1]
        if accepted:
            return {"rects": rects, "coverage": real}

    raise RuntimeError(f"Could not sample an occlusion after {OCC_MAX_ATTEMPTS} attempts.")


def sample_params(condition: int, height: int, width: int, generator: torch.Generator | None = None) -> dict:
    """Sample the parameters of one condition. CLEAN has no parameters."""
    if condition == CLEAN:
        return {}
    if condition == SALT_PEPPER:
        return sample_salt_pepper_params(generator)
    if condition == BLUR:
        return sample_blur_params(generator)
    if condition == OCCLUSION:
        return sample_occlusion_params(height, width, generator)
    raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Dispatchers
# ---------------------------------------------------------------------------

def apply_corruption(img: torch.Tensor, condition: int, params: dict,
                     generator: torch.Generator | None = None) -> torch.Tensor:
    """Apply one condition with the given parameters.

    `generator` is only used by salt-and-pepper (to choose the noisy pixels).
    """
    if condition == CLEAN:
        return img.clone()
    if condition == SALT_PEPPER:
        return salt_and_pepper(img, params["prob"], generator)
    if condition == BLUR:
        return gaussian_blur(img, params["kernel_size"], params["sigma"])
    if condition == OCCLUSION:
        return occlude(img, params["rects"])
    raise ValueError(f"Unknown condition: {condition}")


def corrupt_random(img: torch.Tensor, conditions=(0, 1, 2, 3), generator: torch.Generator | None = None):
    """Pick one condition at random, sample its parameters and apply it.

    Returns (corrupted image, label, params). `label` is a Python int.
    """
    index = _randint(0, len(conditions) - 1, generator)
    label = int(conditions[index])
    _, height, width = img.shape
    params = sample_params(label, height, width, generator)
    corrupted = apply_corruption(img, label, params, generator)
    return corrupted, label, params
