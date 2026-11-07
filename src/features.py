"""Step 2 of the pipeline: turn a note image into a numeric feature vector."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern


def sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low = blurry, high = crisp fine detail."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def edge_density(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 100, 200)
    return float(edges.mean() / 255.0)


def high_freq_ratio(gray: np.ndarray) -> float:
    """Share of spectral energy in high frequencies (fine print / micro-lines)."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    high = mag[dist > min(h, w) * 0.25].sum()
    return float(high / (mag.sum() + 1e-9))


def color_features(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4],
                        [0, 180, 0, 256, 0, 256]).flatten()
    hist /= hist.sum() + 1e-9
    stats = np.concatenate([bgr.reshape(-1, 3).mean(0) / 255.0,
                            bgr.reshape(-1, 3).std(0) / 255.0])
    return np.concatenate([hist, stats])


def texture_features(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    return hist


def extract_features(bgr: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Concatenate colour, edge, texture, sharpness, frequency and HOG features."""
    scalars = np.array([
        np.log1p(sharpness(gray)),
        edge_density(gray),
        high_freq_ratio(gray),
    ])
    hog_vec = hog(gray, pixels_per_cell=(30, 30), cells_per_block=(2, 2),
                  orientations=8, feature_vector=True)
    return np.concatenate([scalars, color_features(bgr),
                           texture_features(gray), hog_vec]).astype(np.float32)
