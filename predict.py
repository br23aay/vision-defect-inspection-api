"""
predict.py
----------
Inference wrapper for the trained MobileNetV2 defect detector.
Loads the saved model once at import time (singleton pattern)
so Flask doesn't reload it on every request.
"""

import json
import os
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH    = "model/defect_model.keras"
CLASS_MAP_PATH = "model/class_map.json"
IMG_SIZE      = (224, 224)

# ── Singleton model loader ────────────────────────────────────────────────────
_model     = None
_class_map = None


def load_model():
    """Load model and class map once, cache globally."""
    global _model, _class_map
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. "
                "Run `python model/train.py` first."
            )
        _model = tf.keras.models.load_model(MODEL_PATH)

    if _class_map is None:
        with open(CLASS_MAP_PATH) as f:
            _class_map = json.load(f)
            # Keys come from JSON as strings — convert to int
            _class_map = {int(k): v for k, v in _class_map.items()}

    return _model, _class_map


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize and normalise a PIL image for MobileNetV2."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # Shape: (1, 224, 224, 3)


def predict(image: Image.Image) -> dict:
    """
    Run inference on a PIL image.

    Returns:
        {
            "label":       "pass" | "fail",
            "confidence":  float (0.0 – 1.0),
            "defect_prob": float,
            "pass_prob":   float,
        }
    """
    model, class_map = load_model()
    tensor = preprocess_image(image)

    raw = float(model.predict(tensor, verbose=0)[0][0])

    # MobileNetV2 output: sigmoid → probability of class index 1
    # Class indices depend on alphabetical folder order: fail=0, pass=1
    # raw = P(class_1) = P(pass)
    pass_prob   = raw
    defect_prob = 1.0 - raw
    class_idx   = 1 if raw >= 0.5 else 0
    label       = class_map[class_idx]
    confidence  = pass_prob if class_idx == 1 else defect_prob

    return {
        "label":       label,
        "confidence":  round(confidence, 4),
        "defect_prob": round(defect_prob, 4),
        "pass_prob":   round(pass_prob, 4),
    }
