"""FastAPI app: validates uploads, preprocesses images, runs the ONNX models.

Run locally (from the repo root):
    uvicorn app.main:app --app-dir backend --port 8000

Endpoints (all under /api):
    GET  /api/health                 which models are loaded
    GET  /api/samples                list of sample images
    GET  /api/samples/{sample_id}    one sample image file
    POST /api/restore/universal      Task 1
    POST /api/restore/hard-routing   Task 2
    POST /api/restore/soft-moe       Task 3
    POST /api/sketch                 Task 4
"""

import logging
from contextlib import asynccontextmanager
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app import config, corruptions, inference
from app.imaging import ImageError, decode_upload, to_png_base64
from app.models import ModelRegistry, ModelUnavailable

logger = logging.getLogger("backend")

SAMPLE_KINDS = {"pets": "pet", "faces": "face"}  # folder name -> kind
SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg"}

Corruption = Literal["none", "salt_pepper", "blur", "occlusion"]
Severity = Literal["low", "medium", "high"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every ONNX model once when the server starts."""
    registry = ModelRegistry(config.MODELS_DIR)
    app.state.registry = registry
    app.state.last_inference_ms = None
    logger.info("Loaded %d of %d models from %s", len(registry.sessions), len(registry.names),
                config.MODELS_DIR)
    yield


app = FastAPI(title="Image Restoration and Sketch API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Errors -> JSON {"detail": ...}
# ---------------------------------------------------------------------------

@app.exception_handler(ImageError)
async def image_error_handler(request: Request, error: ImageError):
    """A bad upload is the user's mistake: 400."""
    return JSONResponse(status_code=400, content={"detail": str(error)})


@app.exception_handler(ModelUnavailable)
async def model_unavailable_handler(request: Request, error: ModelUnavailable):
    """The server works but this model is missing: 503 (service unavailable)."""
    return JSONResponse(status_code=503, content={"detail": str(error)})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def list_samples() -> list:
    """All sample images in SAMPLES_DIR/pets and SAMPLES_DIR/faces (missing folders are skipped)."""
    samples = []
    for folder, kind in SAMPLE_KINDS.items():
        directory = config.SAMPLES_DIR / folder
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in SAMPLE_SUFFIXES:
                sample_id = f"{kind}-{path.name}"
                samples.append({"id": sample_id, "name": path.stem, "kind": kind,
                                "url": f"/api/samples/{sample_id}", "path": path})
    return samples


def find_sample(sample_id: str):
    """Look the id up in the listing. Only listed files can be returned, so "../" tricks fail."""
    for sample in list_samples():
        if sample["id"] == sample_id:
            return sample["path"]
    raise HTTPException(status_code=404, detail=f"Unknown sample: {sample_id}")


def read_upload(file: UploadFile) -> bytes:
    """Check the content type and size of an upload and return its bytes."""
    if file.content_type not in config.ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Upload a PNG or JPG image.")
    # Read one byte more than allowed: if we get it, the file is too big.
    data = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413,
                            detail=f"File too large. The maximum size is {config.MAX_UPLOAD_MB} MB.")
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return data


def load_input(file: UploadFile | None, sample_id: str | None) -> np.ndarray:
    """Get the input image from EITHER an upload OR a sample id, as (3, 128, 128) in [0, 1]."""
    if (file is None) == (sample_id is None):
        raise HTTPException(status_code=400, detail="Send exactly one of 'file' or 'sample_id'.")
    if file is not None:
        return decode_upload(read_upload(file))
    return decode_upload(find_sample(sample_id).read_bytes())


def finish(request: Request, result: dict, input_image: np.ndarray) -> dict:
    """Build the JSON response: images as PNG data URLs plus the other result fields."""
    inference_ms = round(result["inference_ms"], 2)
    request.app.state.last_inference_ms = inference_ms
    response = {key: value for key, value in result.items() if key != "output"}
    response["input_image"] = to_png_base64(input_image)
    response["output_image"] = to_png_base64(result["output"])
    response["inference_ms"] = inference_ms
    return response


def restore(request: Request, operation, file, sample_id, corruption, severity, seed) -> dict:
    """Shared code of the three restoration endpoints."""
    clean = load_input(file, sample_id)
    corrupted, settings = corruptions.apply(clean, corruption, severity, seed)
    result = operation(request.app.state.registry, corrupted)
    response = finish(request, result, corrupted)
    response["corruption"] = settings
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health(request: Request):
    """Server status and which models are loaded."""
    registry = request.app.state.registry
    return {
        "status": "ok",
        "models": registry.status(),
        "loaded_count": len(registry.sessions),
        "expected_count": len(registry.names),
        "last_inference_ms": request.app.state.last_inference_ms,
    }


@app.get("/api/samples")
def samples():
    """The sample images the frontend can offer."""
    return [{key: value for key, value in sample.items() if key != "path"} for sample in list_samples()]


@app.get("/api/samples/{sample_id}")
def sample_file(sample_id: str):
    """Serve one sample image."""
    return FileResponse(find_sample(sample_id))


# The three restoration endpoints take the same form fields.
# Sync functions ("def") run in a worker thread, so a slow model does not block the server.

@app.post("/api/restore/universal")
def restore_universal(request: Request,
                      file: UploadFile | None = File(None),
                      sample_id: str | None = Form(None),
                      corruption: Corruption = Form("none"),
                      severity: Severity = Form("medium"),
                      seed: int | None = Form(None)):
    """Task 1: universal denoising autoencoder."""
    return restore(request, inference.universal, file, sample_id, corruption, severity, seed)


@app.post("/api/restore/hard-routing")
def restore_hard_routing(request: Request,
                         file: UploadFile | None = File(None),
                         sample_id: str | None = Form(None),
                         corruption: Corruption = Form("none"),
                         severity: Severity = Form("medium"),
                         seed: int | None = Form(None)):
    """Task 2: classifier + one specialist (identity for clean images)."""
    return restore(request, inference.hard_routed, file, sample_id, corruption, severity, seed)


@app.post("/api/restore/soft-moe")
def restore_soft_moe(request: Request,
                     file: UploadFile | None = File(None),
                     sample_id: str | None = Form(None),
                     corruption: Corruption = Form("none"),
                     severity: Severity = Form("medium"),
                     seed: int | None = Form(None)):
    """Task 3: soft mixture of experts."""
    return restore(request, inference.soft_moe, file, sample_id, corruption, severity, seed)


@app.post("/api/sketch")
def make_sketch(request: Request,
                file: UploadFile | None = File(None),
                sample_id: str | None = Form(None),
                style: int = Form(1, ge=1, le=3)):
    """Task 4: face photo -> sketch in style 1, 2 or 3."""
    photo = load_input(file, sample_id)
    result = inference.sketch(request.app.state.registry, photo, style)
    return finish(request, result, photo)
