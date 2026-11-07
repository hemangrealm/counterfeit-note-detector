"""Generate a SYNTHETIC demo dataset so the project runs end to end.

These are drawn banknote-like images, NOT real currency. "real" notes get crisp
micro-line patterns; "fake" notes simulate a cheap print/scan (blur, colour shift,
noise, JPEG artefacts). Replace dataset/real and dataset/fake with a real labelled
dataset (e.g. from Kaggle) for meaningful results.

Usage:  python generate_demo_data.py --n 200
"""
import argparse
import random

import cv2
import numpy as np

import config

W, H = 600, 300
PALETTES = [(120, 150, 110), (150, 120, 170), (110, 160, 190), (170, 170, 120), (130, 130, 150)]


def draw_note(rng: random.Random) -> np.ndarray:
    base = np.array(rng.choice(PALETTES), dtype=np.float32) + np.array([rng.randint(-15, 15) for _ in range(3)])
    img = np.ones((H, W, 3), np.float32) * np.clip(base, 0, 255)
    img = img.astype(np.uint8)
    ink = tuple(int(c * 0.45) for c in base)

    # Guilloche-style micro-line pattern (fine, crisp lines)
    cx, cy = rng.randint(380, 480), rng.randint(110, 190)
    for k in range(40):
        pts = []
        r0 = 15 + k * 3
        for t in np.linspace(0, 2 * np.pi, 220):
            r = r0 + 6 * np.sin(8 * t + k * 0.3)
            pts.append([cx + r * np.cos(t), cy + r * np.sin(t) * 0.8])
        cv2.polylines(img, [np.array(pts, np.int32)], True, ink, 1, cv2.LINE_AA)

    # Border lines
    for off in (8, 14, 20):
        cv2.rectangle(img, (off, off), (W - off, H - off), ink, 1, cv2.LINE_AA)

    # Portrait oval with hatching
    cv2.ellipse(img, (140, 150), (60, 85), 0, 0, 360, ink, 2, cv2.LINE_AA)
    for y in range(70, 230, 4):
        cv2.line(img, (95, y), (185, y), ink, 1, cv2.LINE_AA)

    # Security thread with tiny repeating text
    tx = rng.randint(270, 300)
    cv2.rectangle(img, (tx, 0), (tx + 10, H), (40, 40, 40), -1)
    for y in range(12, H, 28):
        cv2.putText(img, "DB", (tx + 1, y), cv2.FONT_HERSHEY_PLAIN, 0.6, (230, 230, 230), 1, cv2.LINE_AA)

    # Text and serial number
    cv2.putText(img, "DEMO BANK", (30, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.9, ink, 2, cv2.LINE_AA)
    cv2.putText(img, str(rng.choice([100, 200, 500, 2000])), (430, 270),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, ink, 2, cv2.LINE_AA)
    serial = f"{rng.randint(1, 9)}AB {rng.randint(100000, 999999)}"
    cv2.putText(img, serial, (30, 280), cv2.FONT_HERSHEY_PLAIN, 1.4, ink, 1, cv2.LINE_AA)
    return img


def jpeg(img, q):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def camera_variation(img, rng):
    """Mild variation both classes get: lighting, tiny rotation, sensor noise."""
    a = rng.uniform(0.85, 1.15)
    b = rng.randint(-15, 15)
    img = cv2.convertScaleAbs(img, alpha=a, beta=b)
    m = cv2.getRotationMatrix2D((W / 2, H / 2), rng.uniform(-2, 2), 1.0)
    img = cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.3:  # slightly soft camera focus, even on genuine notes
        img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.normal(0, rng.uniform(1, 4), img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def make_fake(img, rng):
    """Simulate a low-quality reprint."""
    s = rng.uniform(0.6, 0.97)
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    k = rng.choice([1, 1, 3, 3, 5])
    if k > 1:
        img = cv2.GaussianBlur(img, (k, k), 0)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.85, 1.15)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-6, 6)) % 180
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return jpeg(img, rng.randint(50, 90))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200, help="images per class")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    for sub in ("real", "fake"):
        (config.DATASET_DIR / sub).mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        note = draw_note(rng)
        real = jpeg(camera_variation(note, rng), rng.randint(80, 95))
        fake = make_fake(camera_variation(note, rng), rng)
        cv2.imwrite(str(config.DATASET_DIR / "real" / f"real_{i:04d}.jpg"), real)
        cv2.imwrite(str(config.DATASET_DIR / "fake" / f"fake_{i:04d}.jpg"), fake)
    print(f"Wrote {args.n} real and {args.n} fake demo images to {config.DATASET_DIR}")


if __name__ == "__main__":
    main()
