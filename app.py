"""Flask web app: upload a note photo, get a real/fake prediction."""
from pathlib import Path

from flask import Flask, jsonify, render_template, request

import config
from src.history import add_scan, clear_history, recent_scans
from src.predict import get_detector, load_metrics
from src.preprocess import InvalidImageError

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024


@app.get("/")
def index():
    return render_template("index.html", metrics=load_metrics(), max_mb=config.MAX_UPLOAD_MB)


@app.get("/health")
def health():
    return jsonify(status="ok", model_loaded=config.MODEL_PATH.exists())


@app.post("/api/predict")
def api_predict():
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify(error="Choose an image of a currency note first."), 400
    if Path(file.filename).suffix.lower() not in config.VALID_EXT:
        return jsonify(error="Unsupported file type. Use JPG, PNG, BMP or WebP."), 400
    try:
        # Processed in memory: uploaded images are never written to disk.
        result = get_detector().predict(file.read())
        add_scan(file.filename, result)
    except InvalidImageError as e:
        return jsonify(error=str(e)), 400
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503
    return jsonify(result)


@app.get("/api/history")
def history():
    return jsonify(recent_scans())


@app.delete("/api/history")
def history_clear():
    clear_history()
    return jsonify(status="cleared")


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"Image is larger than {config.MAX_UPLOAD_MB} MB."), 413


if __name__ == "__main__":
    app.run(debug=True)
