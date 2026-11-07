"""Step 1 of the pipeline: load an image and bring it to a standard form."""
from pathlib import Path

import cv2
import numpy as np

import config


class InvalidImageError(ValueError):
    """Raised when the input cannot be decoded as an image."""


def load_image(source) -> np.ndarray:
    """Load a BGR image from a file path or raw bytes."""
    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(str(Path(source)), cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidImageError("File could not be read as an image.")
    return img


def preprocess(img: np.ndarray):
    """Resize, denoise and convert to grayscale.

    Returns (bgr, gray), both at config.IMG_SIZE.
    """
    bgr = cv2.resize(img, config.IMG_SIZE, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Light blur only: heavy blur would erase the fine print we want to measure.
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return bgr, gray
