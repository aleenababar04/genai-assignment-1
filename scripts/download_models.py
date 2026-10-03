"""Download the trained ONNX models into models/.

Run from the repository root with any Python 3.8+ (no extra packages needed):

    python scripts/download_models.py

The files are attached to a GitHub Release of this repository, so they do
not count against Git LFS storage or bandwidth limits.
"""

import sys
import urllib.request
from pathlib import Path

RELEASE_URL = "https://github.com/aleenababar04/genai-assignment-1/releases/download/models-v1"

MODEL_FILES = [
    "task1_udae.onnx",
    "task2_classifier.onnx",
    "task2_specialist_salt_pepper.onnx",
    "task2_specialist_blur.onnx",
    "task2_specialist_occlusion.onnx",
    "task3_moe.onnx",
    "task4_generator.onnx",
]

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def download(file_name):
    """Download one file unless it is already present."""
    target = MODELS_DIR / file_name
    if target.exists() and target.stat().st_size > 0:
        print(f"already present: {file_name}")
        return
    url = f"{RELEASE_URL}/{file_name}"
    print(f"downloading {url}")
    partial = target.with_suffix(".part")
    urllib.request.urlretrieve(url, partial)  # write to a temporary name first
    partial.replace(target)
    print(f"  saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


def main():
    MODELS_DIR.mkdir(exist_ok=True)
    failed = []
    for file_name in MODEL_FILES:
        try:
            download(file_name)
        except Exception as error:  # report every failure, then continue with the rest
            print(f"  FAILED: {error}")
            failed.append(file_name)
    if failed:
        print(f"\n{len(failed)} file(s) could not be downloaded: {', '.join(failed)}")
        print(f"Download them manually from {RELEASE_URL.rsplit('/download', 1)[0]} into {MODELS_DIR}")
        sys.exit(1)
    print("\nall models are in place")


if __name__ == "__main__":
    main()
