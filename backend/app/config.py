"""Settings of the backend. Paths and CORS origins can be changed with environment variables."""

import os
from pathlib import Path

# This file is backend/app/config.py, so parents[1] is backend/ and parents[2] is the repo root.
BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

# Folder with the exported ONNX files (in Docker this is set to /app/models).
MODELS_DIR = Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models"))

# Folder with the sample images (sub-folders "pets" and "faces").
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", BACKEND_DIR / "samples"))

# Upload rules.
MAX_UPLOAD_MB = 10
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg"}

# Every model works on 128x128 RGB images.
IMAGE_SIZE = 128

# Web pages that may call the API (the React dev server, a preview server, the frontend container).
_DEFAULT_ORIGINS = "http://localhost:5173,http://localhost:3000,http://localhost:8080"
CORS_ORIGINS = [origin.strip() for origin in os.environ.get("CORS_ORIGINS", _DEFAULT_ORIGINS).split(",")
                if origin.strip()]
