"""Step 4: load the trained model and classify a single note image."""
import json
from functools import lru_cache

import joblib

import config
from src.features import edge_density, extract_features, sharpness
from src.preprocess import load_image, preprocess


class Detector:
    def __init__(self):
        if not config.MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {config.MODEL_PATH}. Run: python -m src.train")
        self.model = joblib.load(config.MODEL_PATH)

    def predict(self, source) -> dict:
        """source: file path or raw image bytes."""
        bgr, gray = preprocess(load_image(source))
        feats = extract_features(bgr, gray).reshape(1, -1)
        p_real = float(self.model.predict_proba(feats)[0][1])
        sharp = sharpness(gray)

        confidence = max(p_real, 1 - p_real)
        label = "real" if p_real >= 0.5 else "fake"
        if sharp < config.MIN_SHARPNESS:
            status, message = "retake", "Photo is too blurry. Retake it in good light, holding the camera steady."
        elif confidence < config.CONFIDENCE_THRESHOLD:
            status, message = "uncertain", "Result is not confident. Verify this note manually."
        else:
            status, message = "ok", f"Likely {label}."
        return {
            "status": status,
            "label": label,
            "confidence": round(confidence * 100, 1),
            "p_real": round(p_real * 100, 1),
            "message": message,
            "details": {"sharpness": round(sharp, 1), "edge_density": round(edge_density(gray), 4)},
            
        }


@lru_cache(maxsize=1)
def get_detector() -> Detector:
    return Detector()


def load_metrics():
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text())
    return None
