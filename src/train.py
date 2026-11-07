"""Step 3: build features, compare models, evaluate, and save the best one.

Usage:  python -m src.train            (reuses cached features if present)
        python -m src.train --rebuild  (re-extract features from images)
"""
import argparse
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)

import config
from src.features import extract_features
from src.preprocess import InvalidImageError, load_image, preprocess


def build_feature_table() -> pd.DataFrame:
    rows = []
    for label, folder in config.CLASS_NAMES.items():
        folder_path = config.DATASET_DIR / folder
        files = [f for f in sorted(folder_path.glob("*")) if f.suffix.lower() in config.VALID_EXT]
        if not files:
            raise SystemExit(f"No images found in {folder_path}. "
                             "Add images or run: python generate_demo_data.py")
        for f in files:
            try:
                bgr, gray = preprocess(load_image(f))
            except InvalidImageError:
                print(f"Skipping unreadable file: {f.name}")
                continue
            rows.append([f.name, label, *extract_features(bgr, gray)])
    cols = ["file", "label"] + [f"f{i}" for i in range(len(rows[0]) - 2)]
    return pd.DataFrame(rows, columns=cols)


def get_table(rebuild: bool) -> pd.DataFrame:
    if config.FEATURE_CACHE.exists() and not rebuild:
        print("Using cached features:", config.FEATURE_CACHE)
        return pd.read_csv(config.FEATURE_CACHE)
    print("Extracting features from images...")
    df = build_feature_table()
    config.FEATURE_CACHE.parent.mkdir(exist_ok=True)
    df.to_csv(config.FEATURE_CACHE, index=False)
    return df


def candidates():
    rs = config.RANDOM_STATE
    return {
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "SVM (RBF)": SVC(probability=True, C=3, random_state=rs),
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=rs),
        "Gradient Boosting": GradientBoostingClassifier(random_state=rs),
    }


def main(rebuild: bool = False):
    config.REPORTS_DIR.mkdir(exist_ok=True)
    config.MODEL_PATH.parent.mkdir(exist_ok=True)

    df = get_table(rebuild)
    X = df.drop(columns=["file", "label"]).values
    y = df["label"].values
    print(f"Dataset: {len(y)} images ({(y == 1).sum()} real, {(y == 0).sum()} fake), "
          f"{X.shape[1]} features")

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    # Compare models with 5-fold cross-validation on the training split only
    cv = StratifiedKFold(5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = {}
    for name, clf in candidates().items():
        pipe = Pipeline([("scale", StandardScaler()), ("clf", clf)])
        s = cross_val_score(pipe, Xtr, ytr, cv=cv, scoring="f1")
        scores[name] = float(s.mean())
        print(f"  {name:<20} CV F1 = {s.mean():.3f} (+/- {s.std():.3f})")

    best_name = max(scores, key=scores.get)
    print("Best model:", best_name)
    best = Pipeline([("scale", StandardScaler()), ("clf", candidates()[best_name])])
    best.fit(Xtr, ytr)

    pred = best.predict(Xte)
    proba = best.predict_proba(Xte)[:, 1]
    metrics = {
        "best_model": best_name,
        "cv_f1": scores,
        "test_accuracy": float(accuracy_score(yte, pred)),
        "test_precision": float(precision_score(yte, pred)),
        "test_recall": float(recall_score(yte, pred)),
        "test_f1": float(f1_score(yte, pred)),
        "test_roc_auc": float(roc_auc_score(yte, proba)),
        "n_train": int(len(ytr)),
        "n_test": int(len(yte)),
        "dataset_note": "demo/synthetic" if (config.DATASET_DIR / "real" / "real_0000.jpg").exists() else "custom",
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                      for k, v in metrics.items() if k.startswith("test_")}, indent=2))

    # Plots for the report
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        yte, pred, display_labels=["fake", "real"], cmap="Greens", ax=ax, colorbar=False)
    ax.set_title("Confusion matrix (test set)")
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(yte, proba, ax=ax, name=best_name)
    ax.set_title("ROC curve (test set)")
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "roc_curve.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(list(scores), list(scores.values()), color="#0F6B5B")
    ax.set_xlim(max(0, min(scores.values()) - 0.1), 1.0)
    ax.set_xlabel("5-fold CV F1"); ax.set_title("Model comparison")
    fig.tight_layout(); fig.savefig(config.REPORTS_DIR / "model_comparison.png", dpi=150); plt.close(fig)

    joblib.dump(best, config.MODEL_PATH)
    print("Saved model ->", config.MODEL_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="re-extract features")
    main(ap.parse_args().rebuild)
