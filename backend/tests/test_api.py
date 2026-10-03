"""End-to-end tests of the HTTP API with dummy ONNX models."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from conftest import DEFAULT_CLASS, make_png_bytes

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")
pytest.importorskip("cv2")

from app.models import EXPECTED_MODELS  # noqa: E402

RESTORE_ENDPOINTS = {
    "/api/restore/universal": [],
    "/api/restore/hard-routing": ["probabilities", "predicted", "selected_expert", "identity_bypass"],
    "/api/restore/soft-moe": ["weights", "top_branch"],
}
UNSUPPORTED = "Unsupported file type. Upload a PNG or JPG image."


def upload(data=None, name="image.png", content_type="image/png"):
    """The `files` argument for an upload."""
    return {"file": (name, data if data is not None else make_png_bytes(), content_type)}


def decode(data_url: str) -> np.ndarray:
    """Data URL -> uint8 array (H, W, 3)."""
    assert data_url.startswith("data:image/png;base64,")
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))).convert("RGB"))


# ---------------------------------------------------------------------------
# Health and samples
# ---------------------------------------------------------------------------

def test_health_all_loaded(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert list(body["models"]) == EXPECTED_MODELS
    assert all(model["loaded"] for model in body["models"].values())
    assert body["loaded_count"] == body["expected_count"] == len(EXPECTED_MODELS)
    assert body["last_inference_ms"] is None


def test_health_reports_missing_model(make_client, models_dir_without_moe):
    body = make_client(models_dir_without_moe).get("/api/health").json()
    assert body["models"]["task3_moe.onnx"]["loaded"] is False
    assert "not found" in body["models"]["task3_moe.onnx"]["error"]
    assert body["models"]["task1_udae.onnx"] == {"loaded": True, "error": None}
    assert body["loaded_count"] == len(EXPECTED_MODELS) - 1


def test_last_inference_ms_is_recorded(client):
    client.post("/api/restore/universal", files=upload())
    assert isinstance(client.get("/api/health").json()["last_inference_ms"], float)


def test_samples_listing_and_file(client):
    samples = client.get("/api/samples").json()
    assert {(s["id"], s["kind"]) for s in samples} == {("pet-cat.png", "pet"), ("face-person.jpg", "face")}
    for sample in samples:
        assert set(sample) == {"id", "name", "kind", "url"}
        response = client.get(sample["url"])
        assert response.status_code == 200
        assert response.content[:4] in (b"\x89PNG", b"\xff\xd8\xff\xe0", b"\xff\xd8\xff\xdb")


@pytest.mark.parametrize("bad_id", ["../secret.png", "..%2Fsecret.png", "pet-..%2F..%2Fsecret.png",
                                    "pet-notes.txt", "secret.png", "pet-missing.png"])
def test_sample_path_traversal_is_rejected(client, bad_id):
    assert client.get(f"/api/samples/{bad_id}").status_code == 404


def test_sample_id_traversal_in_form_is_rejected(client):
    response = client.post("/api/restore/universal", data={"sample_id": "../secret.png"})
    assert response.status_code == 404


def test_missing_samples_folder_gives_empty_list(client, monkeypatch, tmp_path):
    from app import config
    monkeypatch.setattr(config, "SAMPLES_DIR", tmp_path / "does_not_exist")
    assert client.get("/api/samples").json() == []


# ---------------------------------------------------------------------------
# Restoration endpoints
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint", list(RESTORE_ENDPOINTS))
@pytest.mark.parametrize("source", ["upload", "sample"])
def test_restore_endpoints(client, endpoint, source):
    if source == "upload":
        response = client.post(endpoint, files=upload(), data={"corruption": "blur", "severity": "low"})
    else:
        response = client.post(endpoint, data={"sample_id": "pet-cat.png", "corruption": "blur",
                                               "severity": "low"})
    assert response.status_code == 200, response.text
    body = response.json()
    for key in ["input_image", "output_image", "corruption", "inference_ms", *RESTORE_ENDPOINTS[endpoint]]:
        assert key in body, key
    assert "output" not in body
    assert decode(body["input_image"]).shape == decode(body["output_image"]).shape == (128, 128, 3)
    assert body["corruption"] == {"type": "blur", "severity": "low", "kernel_size": 3, "sigma": 0.7,
                                  "seed": body["corruption"]["seed"]}
    assert body["inference_ms"] >= 0


@pytest.mark.parametrize("corruption", ["salt_pepper", "blur", "occlusion"])
def test_corruption_changes_the_input(client, corruption):
    clean = client.post("/api/restore/universal", files=upload()).json()
    assert clean["corruption"] == {"type": "none"}
    body = client.post("/api/restore/universal", files=upload(),
                       data={"corruption": corruption, "severity": "high", "seed": "5"}).json()
    settings = body["corruption"]
    assert settings["type"] == corruption and settings["severity"] == "high" and settings["seed"] == 5
    if corruption == "salt_pepper":
        assert settings["probability"] == 0.15
    if corruption == "occlusion":
        assert len(settings["rectangles"]) == 3 and abs(settings["coverage"] - 0.35) <= 0.01
    assert not np.array_equal(decode(body["input_image"]), decode(clean["input_image"]))


def test_seed_makes_api_corruption_reproducible(client):
    def corrupted(seed):
        data = {"corruption": "salt_pepper", "severity": "medium"}
        if seed is not None:
            data["seed"] = str(seed)
        body = client.post("/api/restore/universal", files=upload(), data=data).json()
        return decode(body["input_image"]), body["corruption"]["seed"]

    first, _ = corrupted(42)
    second, _ = corrupted(42)
    other, _ = corrupted(43)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)
    # Without a seed the server picks one and returns it; sending it back reproduces the image.
    random_result, drawn_seed = corrupted(None)
    assert np.array_equal(random_result, corrupted(drawn_seed)[0])


def test_hard_routing_clean_returns_input(make_client, models_dir_for):
    client = make_client(models_dir_for("clean"))
    body = client.post("/api/restore/hard-routing", files=upload()).json()
    assert body["predicted"] == "clean"
    assert body["selected_expert"] == "identity"
    assert body["identity_bypass"] is True
    assert np.array_equal(decode(body["output_image"]), decode(body["input_image"]))


def test_hard_routing_non_clean_uses_expert(client):
    # The default dummy classifier predicts "blur"; the dummy blur specialist computes 1 - x.
    body = client.post("/api/restore/hard-routing", files=upload(),
                       data={"corruption": "blur", "severity": "medium"}).json()
    assert body["predicted"] == DEFAULT_CLASS == "blur"
    assert body["selected_expert"] == "blur"
    assert body["identity_bypass"] is False
    assert set(body["probabilities"]) == {"clean", "salt_pepper", "blur", "occlusion"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    inp = decode(body["input_image"]).astype(int)
    out = decode(body["output_image"]).astype(int)
    assert np.abs(out - (255 - inp)).max() <= 1


def test_soft_moe_weights_sum_to_one(client):
    body = client.post("/api/restore/soft-moe", files=upload(), data={"corruption": "occlusion"}).json()
    assert set(body["weights"]) == {"identity", "salt_pepper", "blur", "occlusion"}
    assert abs(sum(body["weights"].values()) - 1.0) < 1e-5
    assert body["top_branch"] == max(body["weights"], key=body["weights"].get)


def test_missing_model_gives_503(make_client, models_dir_without_moe):
    response = make_client(models_dir_without_moe).post("/api/restore/soft-moe", files=upload())
    assert response.status_code == 503
    assert response.json() == {"detail": "Model task3_moe.onnx is not loaded."}


def test_invalid_corruption_or_severity(client):
    assert client.post("/api/restore/universal", files=upload(), data={"corruption": "fog"}).status_code == 422
    assert client.post("/api/restore/universal", files=upload(), data={"severity": "extreme"}).status_code == 422


# ---------------------------------------------------------------------------
# Sketch endpoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("style", [1, 2, 3])
def test_sketch_styles(client, style):
    response = client.post("/api/sketch", files=upload(), data={"style": str(style)})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"input_image", "output_image", "style", "inference_ms"}
    assert body["style"] == style
    assert decode(body["output_image"]).shape == (128, 128, 3)  # 1-channel sketch shown as RGB


def test_sketch_from_sample(client):
    response = client.post("/api/sketch", data={"sample_id": "face-person.jpg", "style": "2"})
    assert response.status_code == 200


@pytest.mark.parametrize("style", ["0", "4", "abc"])
def test_sketch_rejects_bad_style(client, style):
    assert client.post("/api/sketch", files=upload(), data={"style": style}).status_code in (400, 422)


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("endpoint", [*RESTORE_ENDPOINTS, "/api/sketch"])
def test_neither_or_both_inputs_rejected(client, endpoint):
    neither = client.post(endpoint, data={"corruption": "none"} if "restore" in endpoint else {"style": "1"})
    both = client.post(endpoint, files=upload(), data={"sample_id": "pet-cat.png"})
    for response in (neither, both):
        assert response.status_code == 400
        assert response.json() == {"detail": "Send exactly one of 'file' or 'sample_id'."}


@pytest.mark.parametrize("name, data, content_type", [
    ("doc.pdf", b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\n", "application/pdf"),
    ("notes.txt", b"hello world", "text/plain"),
])
def test_wrong_content_type_rejected(client, name, data, content_type):
    response = client.post("/api/restore/universal", files=upload(data, name, content_type))
    assert response.status_code == 400
    assert response.json() == {"detail": UNSUPPORTED}


def test_lying_content_type_rejected(client):
    gif = io.BytesIO()
    Image.new("RGB", (16, 16)).save(gif, format="GIF")
    response = client.post("/api/restore/universal", files=upload(gif.getvalue(), "fake.png", "image/png"))
    assert response.status_code == 400
    assert response.json() == {"detail": UNSUPPORTED}

    pdf_as_png = upload(b"%PDF-1.4 not really an image", "fake.png", "image/png")
    response = client.post("/api/restore/universal", files=pdf_as_png)
    assert response.status_code == 400
    assert "Could not read the image" in response.json()["detail"]


def test_oversized_upload_rejected(client):
    from app import config
    data = make_png_bytes() + b"\0" * (config.MAX_UPLOAD_BYTES + 1)
    response = client.post("/api/restore/universal", files=upload(data))
    assert response.status_code == 413
    assert response.json() == {"detail": "File too large. The maximum size is 10 MB."}


def test_empty_upload_rejected(client):
    response = client.post("/api/restore/universal", files=upload(b""))
    assert response.status_code == 400


def test_jpeg_upload_accepted(client):
    jpeg = io.BytesIO()
    Image.open(io.BytesIO(make_png_bytes())).save(jpeg, format="JPEG")
    response = client.post("/api/restore/universal", files=upload(jpeg.getvalue(), "photo.jpg", "image/jpeg"))
    assert response.status_code == 200
