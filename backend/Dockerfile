# Inference-only image for the FastAPI backend (no PyTorch).
#
# Build context: the backend/ folder.
#   docker build -t restoration-backend backend
# The ONNX files are NOT copied into the image; mount the repo's models/ folder:
#   docker run -p 8000:8000 -v "$(pwd)/models:/app/models:ro" restoration-backend
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/app/models \
    SAMPLES_DIR=/app/samples

WORKDIR /app

# No apt packages are needed: opencv-python-headless has no GUI, so it does not need libGL.
# Install the dependencies first so this layer is cached when only the code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY samples ./samples

# Mount point for the ONNX models.
RUN mkdir -p /app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health', timeout=4)" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
