"""Loading the ONNX models once and running them.

Missing or broken files do not stop the server: they are recorded, shown by
the health endpoint, and only the endpoints that need them fail (HTTP 503).
"""

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Every ONNX file the app knows about (written by src/export/).
EXPECTED_MODELS = [
    "task1_udae.onnx",
    "task2_classifier.onnx",
    "task2_specialist_salt_pepper.onnx",
    "task2_specialist_blur.onnx",
    "task2_specialist_occlusion.onnx",
    "task3_moe.onnx",
    "task4_generator.onnx",
]


class ModelUnavailable(RuntimeError):
    """A model was asked for but is not loaded."""

    def __init__(self, name: str):
        super().__init__(f"Model {name} is not loaded.")
        self.name = name


class ModelRegistry:
    """Holds one ONNX Runtime session per loaded model."""

    def __init__(self, models_dir: Path, names: list | None = None):
        self.models_dir = Path(models_dir)
        self.names = list(names or EXPECTED_MODELS)
        self.sessions = {}  # name -> InferenceSession
        self.errors = {}  # name -> error text, for missing or broken files
        self.load_all()

    def load_all(self) -> None:
        """Try to load every expected model and record what happened."""
        for name in self.names:
            path = self.models_dir / name
            if not path.is_file():
                self.errors[name] = f"File not found: {path}"
                continue
            try:
                self.sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            except Exception as error:  # a corrupt or incompatible file
                self.errors[name] = f"Failed to load: {error}"

    def get(self, name: str) -> ort.InferenceSession:
        """Return the session for `name`, or raise ModelUnavailable."""
        if name not in self.sessions:
            raise ModelUnavailable(name)
        return self.sessions[name]

    def run(self, name: str, feeds: dict) -> tuple:
        """Run model `name` on `feeds` ({input name: array}).

        Returns ({output name: array}, time in milliseconds).
        """
        session = self.get(name)
        output_names = [output.name for output in session.get_outputs()]
        start = time.perf_counter()
        values = session.run(output_names, feeds)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return dict(zip(output_names, values)), elapsed_ms

    def status(self) -> dict:
        """{name: {"loaded": bool, "error": str or None}} for the health endpoint."""
        return {name: {"loaded": name in self.sessions, "error": self.errors.get(name)}
                for name in self.names}


def as_batch(image: np.ndarray) -> np.ndarray:
    """(3, H, W) -> (1, 3, H, W) float32, the shape every model expects."""
    return np.ascontiguousarray(image[None], dtype=np.float32)
