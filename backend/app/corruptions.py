"""NumPy/OpenCV version of the corruptions in src/data/corruptions.py.

The backend has no PyTorch, so the three corruptions are re-written here with
the SAME definitions and the FIXED test severities used for evaluation:

  salt_pepper: each pixel (all 3 channels together) becomes white or black
               with probability p; half of the changed pixels are white.
  blur:        Gaussian blur with a square kernel (k x k) and standard deviation sigma.
  occlusion:   n non-overlapping black rectangles covering a target fraction
               of the image (within 0.01), with random positions and shapes.

Images are float32 arrays (3, H, W) in [0, 1]. Inputs are never changed.
"""

import cv2
import numpy as np

CORRUPTIONS = ["none", "salt_pepper", "blur", "occlusion"]
SEVERITIES = ["low", "medium", "high"]

# Fixed test severities (copied from src/data/corruptions.py).
TEST_SP_PROB = {"low": 0.03, "medium": 0.08, "high": 0.15}
TEST_BLUR = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}  # (kernel_size, sigma)
TEST_OCCLUSION = {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}  # (num_rects, coverage)

# Settings of the occlusion sampler (copied from src/data/corruptions.py).
OCC_SHARE_RANGE = (0.5, 1.5)  # relative area of each rectangle before normalising
OCC_ASPECT_RANGE = (0.5, 2.0)  # height / width of a rectangle
OCC_COVERAGE_TOLERANCE = 0.01  # allowed error from the target coverage
OCC_PLACEMENT_TRIES = 50  # tries to place one rectangle without overlap
OCC_MAX_ATTEMPTS = 1000  # tries to build one complete set of rectangles


# ---------------------------------------------------------------------------
# The three corruptions
# ---------------------------------------------------------------------------

def salt_and_pepper(img: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Turn each pixel white or black with probability `prob` (all channels together)."""
    _, height, width = img.shape
    selected = rng.random((height, width)) < prob  # which pixels are corrupted
    is_salt = rng.random((height, width)) < 0.5  # white (True) or black (False)

    out = img.copy()
    out[:, selected & is_salt] = 1.0  # the (H, W) mask is applied to every channel
    out[:, selected & ~is_salt] = 0.0
    return out


def gaussian_blur(img: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
    """Blur with a k x k Gaussian kernel.

    Matches torchvision's gaussian_blur: a normalised Gaussian kernel and
    reflect padding (OpenCV calls it BORDER_REFLECT_101).
    """
    hwc = np.ascontiguousarray(img.transpose(1, 2, 0), dtype=np.float32)  # OpenCV wants (H, W, C)
    blurred = cv2.GaussianBlur(hwc, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma,
                               borderType=cv2.BORDER_REFLECT_101)
    return np.clip(blurred.transpose(2, 0, 1), 0.0, 1.0).astype(np.float32)


def occlude(img: np.ndarray, rects: list) -> np.ndarray:
    """Paint black rectangles. `rects` is a list of [top, left, height, width]."""
    out = img.copy()
    for top, left, rect_h, rect_w in rects:
        out[:, top:top + rect_h, left:left + rect_w] = 0.0
    return out


def coverage_of(rects: list, height: int, width: int) -> float:
    """Fraction of the image covered by the rectangles (overlaps counted once)."""
    mask = np.zeros((height, width), dtype=bool)
    for top, left, rect_h, rect_w in rects:
        mask[top:top + rect_h, left:left + rect_w] = True
    return float(mask.sum()) / (height * width)


# ---------------------------------------------------------------------------
# Occlusion sampler (same algorithm as src/data/corruptions.py)
# ---------------------------------------------------------------------------

def _uniform(low: float, high: float, rng: np.random.Generator) -> float:
    """One float drawn uniformly from [low, high)."""
    return low + (high - low) * float(rng.random())


def _randint(low: int, high: int, rng: np.random.Generator) -> int:
    """One int drawn uniformly from low..high (both inclusive)."""
    return int(rng.integers(low, high + 1))


def _rects_overlap(a: list, b: list) -> bool:
    """True if rectangles a and b ([top, left, height, width]) share any pixel."""
    a_top, a_left, a_h, a_w = a
    b_top, b_left, b_h, b_w = b
    separate_vertically = a_top + a_h <= b_top or b_top + b_h <= a_top
    separate_horizontally = a_left + a_w <= b_left or b_left + b_w <= a_left
    return not (separate_vertically or separate_horizontally)


def _try_build_rects(height: int, width: int, num_rects: int, coverage: float,
                     rng: np.random.Generator):
    """One attempt at `num_rects` non-overlapping rectangles covering about `coverage`.

    Returns the rectangles, or None if one of them could not be placed.
    """
    # 1. Split the target area between the rectangles with random shares.
    shares = [_uniform(OCC_SHARE_RANGE[0], OCC_SHARE_RANGE[1], rng) for _ in range(num_rects)]
    total_share = sum(shares)
    target_area = coverage * height * width

    rects = []
    for share in shares:
        area = target_area * share / total_share

        # 2. Pick a shape: aspect = h / w and area = h * w give h and w.
        aspect = _uniform(OCC_ASPECT_RANGE[0], OCC_ASPECT_RANGE[1], rng)
        rect_h = max(1, min(round((area * aspect) ** 0.5), height))
        rect_w = max(1, min(round((area / aspect) ** 0.5), width))

        # 3. Try random positions until the rectangle does not hit the others.
        placed = False
        for _ in range(OCC_PLACEMENT_TRIES):
            top = _randint(0, height - rect_h, rng)
            left = _randint(0, width - rect_w, rng)
            candidate = [top, left, rect_h, rect_w]
            if not any(_rects_overlap(candidate, other) for other in rects):
                rects.append(candidate)
                placed = True
                break
        if not placed:
            return None
    return rects


def sample_occlusion(height: int, width: int, num_rects: int, coverage: float,
                     rng: np.random.Generator) -> tuple:
    """Sample rectangles until their real coverage is within the tolerance of `coverage`.

    Returns (rects, real coverage).
    """
    for _ in range(OCC_MAX_ATTEMPTS):
        rects = _try_build_rects(height, width, num_rects, coverage, rng)
        if rects is None:
            continue  # a rectangle could not be placed: start again
        real = coverage_of(rects, height, width)
        if abs(real - coverage) <= OCC_COVERAGE_TOLERANCE:
            return rects, real
    raise RuntimeError(f"Could not sample an occlusion after {OCC_MAX_ATTEMPTS} attempts.")


# ---------------------------------------------------------------------------
# Entry point used by the API
# ---------------------------------------------------------------------------

def apply(image: np.ndarray, corruption: str, severity: str, seed: int | None = None):
    """Corrupt `image` with a fixed test severity.

    Returns (corrupted image, settings). `settings` is a JSON-friendly dict
    that describes exactly what was done, including the seed, so the same
    corruption can be reproduced by sending the seed again.
    """
    if corruption not in CORRUPTIONS:
        raise ValueError(f"Unknown corruption: {corruption!r}")
    if corruption == "none":
        return image.copy(), {"type": "none"}
    if severity not in SEVERITIES:
        raise ValueError(f"Unknown severity: {severity!r}")

    # No seed given: draw one, so the result can still be reproduced later.
    if seed is None:
        seed = int(np.random.default_rng().integers(0, 2**31 - 1))
    rng = np.random.default_rng(seed)
    settings = {"type": corruption, "severity": severity}

    if corruption == "salt_pepper":
        prob = TEST_SP_PROB[severity]
        corrupted = salt_and_pepper(image, prob, rng)
        settings["probability"] = prob
    elif corruption == "blur":
        kernel_size, sigma = TEST_BLUR[severity]
        corrupted = gaussian_blur(image, kernel_size, sigma)
        settings["kernel_size"] = kernel_size
        settings["sigma"] = sigma
    else:  # occlusion
        num_rects, coverage = TEST_OCCLUSION[severity]
        _, height, width = image.shape
        rects, real = sample_occlusion(height, width, num_rects, coverage, rng)
        corrupted = occlude(image, rects)
        settings["num_rects"] = num_rects
        settings["target_coverage"] = coverage
        settings["coverage"] = round(real, 4)
        settings["rectangles"] = rects

    settings["seed"] = seed
    return corrupted, settings
