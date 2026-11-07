"""Central configuration for the Fake Currency Detection project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "dataset"         
MODEL_PATH = ROOT / "models" / "currency_model.joblib"
FEATURE_CACHE = ROOT / "data_cache" / "features.csv"
REPORTS_DIR = ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"

IMG_SIZE = (300, 150)                   
CLASS_NAMES = {0: "fake", 1: "real"}    
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

RANDOM_STATE = 42
TEST_SIZE = 0.2


MAX_UPLOAD_MB = 5

CONFIDENCE_THRESHOLD = 0.70
MIN_SHARPNESS = 15.0
AUGMENT_COPIES = 3


REGIONS = {
    "thread":   (0.43, 0.00, 0.53, 1.00),
    "portrait": (0.13, 0.20, 0.33, 0.80),
    "serial":   (0.03, 0.85, 0.40, 1.00),
    "value":    (0.70, 0.82, 0.97, 0.98),
}

HISTORY_DB = ROOT / "data_cache" / "history.db"
HISTORY_LIMIT = 10

CNN_PATH = ROOT / "models" / "cnn_model.pt"
CNN_METRICS_PATH = REPORTS_DIR / "cnn_metrics.json"
