"""The four operations of the app, as plain functions on NumPy arrays.

Every function takes one image (3, 128, 128) float32 in [0, 1] and returns a
dict with the output image (C, 128, 128), extra routing information and the
model time in milliseconds.
"""

import numpy as np

from app.models import ModelRegistry, as_batch

# Classifier classes and soft-mixture branches, in index order.
CONDITION_NAMES = ["clean", "salt_pepper", "blur", "occlusion"]
BRANCH_NAMES = ["identity", "salt_pepper", "blur", "occlusion"]

# Specialist file used for each non-clean class.
SPECIALIST_FILES = {
    "salt_pepper": "task2_specialist_salt_pepper.onnx",
    "blur": "task2_specialist_blur.onnx",
    "occlusion": "task2_specialist_occlusion.onnx",
}


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax of a 1-D array (the max is subtracted for numerical safety)."""
    shifted = np.exp(logits - logits.max())
    return shifted / shifted.sum()


def universal(registry: ModelRegistry, image: np.ndarray) -> dict:
    """Task 1: one denoising autoencoder for every corruption."""
    outputs, elapsed_ms = registry.run("task1_udae.onnx", {"input": as_batch(image)})
    return {"output": outputs["output"][0], "inference_ms": elapsed_ms}


def hard_routed(registry: ModelRegistry, image: np.ndarray) -> dict:
    """Task 2: the classifier picks ONE specialist; "clean" returns the input unchanged."""
    outputs, classifier_ms = registry.run("task2_classifier.onnx", {"input": as_batch(image)})
    probs = softmax(outputs["logits"][0].astype(np.float64))
    predicted = CONDITION_NAMES[int(probs.argmax())]

    if predicted == "clean":
        # Identity bypass: no specialist is run.
        output, expert, specialist_ms = image.copy(), "identity", 0.0
    else:
        # Only the chosen specialist runs (a missing file raises ModelUnavailable -> 503).
        specialist_outputs, specialist_ms = registry.run(SPECIALIST_FILES[predicted],
                                                         {"input": as_batch(image)})
        output, expert = specialist_outputs["output"][0], predicted

    return {
        "output": output,
        "probabilities": {name: float(p) for name, p in zip(CONDITION_NAMES, probs)},
        "predicted": predicted,
        "selected_expert": expert,
        "identity_bypass": predicted == "clean",
        "inference_ms": classifier_ms + specialist_ms,
    }


def soft_moe(registry: ModelRegistry, image: np.ndarray) -> dict:
    """Task 3: the gate mixes all four branches; the weights come out of the model."""
    outputs, elapsed_ms = registry.run("task3_moe.onnx", {"input": as_batch(image)})
    weights = outputs["weights"][0]
    return {
        "output": outputs["output"][0],
        "weights": {name: float(w) for name, w in zip(BRANCH_NAMES, weights)},
        "top_branch": BRANCH_NAMES[int(weights.argmax())],
        "inference_ms": elapsed_ms,
    }


def sketch(registry: ModelRegistry, photo: np.ndarray, style: int) -> dict:
    """Task 4: turn a face photo into a sketch. `style` is 1, 2 or 3 (the model uses 0, 1, 2)."""
    if style not in (1, 2, 3):
        raise ValueError("style must be 1, 2 or 3")
    feeds = {"photo": as_batch(photo), "style": np.array([style - 1], dtype=np.int64)}
    outputs, elapsed_ms = registry.run("task4_generator.onnx", feeds)
    return {"output": outputs["sketch"][0], "style": style, "inference_ms": elapsed_ms}
