import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from src.features import extract_features
from src.preprocess import InvalidImageError, load_image, preprocess


def sample_bytes(color=(120, 150, 110)):
    img = np.full((300, 600, 3), color, np.uint8)
    cv2.putText(img, "TEST", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 3, (20, 20, 20), 3)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def test_preprocess_size():
    bgr, gray = preprocess(load_image(sample_bytes()))
    assert bgr.shape[:2] == (config.IMG_SIZE[1], config.IMG_SIZE[0])
    assert gray.shape == (config.IMG_SIZE[1], config.IMG_SIZE[0])


def test_feature_vector_is_stable_and_finite():
    bgr, gray = preprocess(load_image(sample_bytes()))
    a, b = extract_features(bgr, gray), extract_features(bgr, gray)
    assert a.ndim == 1 and np.isfinite(a).all()
    assert np.array_equal(a, b)


def test_invalid_image_rejected():
    with pytest.raises(InvalidImageError):
        load_image(b"not an image")


@pytest.fixture
def client():
    from app import app
    app.config["TESTING"] = True
    return app.test_client()


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"

def test_index_page(client):
    assert client.get("/").status_code == 200

def test_predict_requires_file(client):
    assert client.post("/api/predict").status_code == 400

def test_predict_rejects_bad_extension(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(b"x"), "a.txt")},
                    content_type="multipart/form-data")
    assert r.status_code == 400

def test_predict_rejects_corrupt_image(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(b"junk"), "a.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400

@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="model not trained yet")
def test_predict_returns_verdict(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(sample_bytes()), "n.jpg")},
                    content_type="multipart/form-data")
    body = r.get_json()
    assert r.status_code == 200
    assert body["label"] in ("real", "fake")
    assert 50 <= body["confidence"] <= 100

def test_augment_keeps_shape_and_changes_pixels():
    from src.augment import augment
    bgr, _ = preprocess(load_image(sample_bytes()))
    out = augment(bgr, np.random.default_rng(0))
    assert out.shape == bgr.shape and out.dtype == np.uint8
    assert not np.array_equal(out, bgr)

def test_region_stats_cover_every_configured_region():
    from src.features import region_stats
    _, gray = preprocess(load_image(sample_bytes()))
    stats = region_stats(gray)
    assert set(stats) == set(config.REGIONS)


def test_history_roundtrip():
    import tempfile
    from src import history
    original = config.HISTORY_DB
    config.HISTORY_DB = Path(tempfile.mkdtemp()) / "h.db"
    try:
        history.add_scan("a.jpg", {"status": "ok", "label": "real", "confidence": 97.5})
        rows = history.recent_scans()
        assert rows[0]["filename"] == "a.jpg" and rows[0]["label"] == "real"
        history.clear_history()
        assert history.recent_scans() == []
    finally:
        config.HISTORY_DB = original
