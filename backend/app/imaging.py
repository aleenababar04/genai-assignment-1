"""Turning uploaded bytes into model input arrays, and model outputs into PNG data URLs.

Arrays are float32 with shape (C, H, W) and values in [0, 1], like the
tensors used in training.
"""

import base64
import io

import numpy as np
from PIL import Image

from app import config

# Formats we accept, as named by Pillow after it has read the file.
ALLOWED_FORMATS = {"PNG", "JPEG"}


class ImageError(ValueError):
    """The uploaded data is not an image we can use. The message is shown to the user."""


def decode_upload(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG bytes into a float32 array (3, 128, 128) in [0, 1].

    Preprocessing is identical to `load_image` in src/data/prepare_pet.py:
    convert to RGB, then resize directly to 128x128 with bicubic resampling.
    """
    try:
        with Image.open(io.BytesIO(data)) as image:
            # The file extension and content type can lie; Pillow checks the real format.
            if image.format not in ALLOWED_FORMATS:
                raise ImageError("Unsupported file type. Upload a PNG or JPG image.")
            image = image.convert("RGB")  # greyscale / alpha -> 3 channels
            image = image.resize((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.BICUBIC)
            pixels = np.asarray(image, dtype=np.uint8)  # (H, W, 3)
    except ImageError:
        raise
    except Exception as error:  # Pillow raises several error types for broken files
        raise ImageError("Could not read the image. Upload a valid PNG or JPG file.") from error

    # (H, W, 3) uint8 -> (3, H, W) float in [0, 1], the same as in training.
    return pixels.transpose(2, 0, 1).astype(np.float32) / 255.0


def to_png_base64(array: np.ndarray) -> str:
    """Encode a (3, H, W) or (1, H, W) array in [0, 1] as a "data:image/png;base64,..." URL."""
    if array.shape[0] == 1:
        array = np.repeat(array, 3, axis=0)  # grey -> RGB
    # Round to the nearest 8-bit value and go back to (H, W, 3).
    pixels = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8).transpose(1, 2, 0)

    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")  # uint8 (H, W, 3) is read as RGB
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
