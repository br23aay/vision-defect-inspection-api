"""
app.py
------
Flask REST API — Vision-Based Defect & Quality Inspection
Swayam Ltd · ML Engineering · Production Service

Endpoints:
    POST /predict       Upload an image → get PASS/FAIL prediction
    GET  /logs          Query recent prediction logs from SQLite
    GET  /stats         Aggregate statistics across all predictions
    GET  /health        Health check (model loaded, DB connected)
    GET  /              API info and endpoint reference

Run locally:
    python app.py

Run with gunicorn (production):
    gunicorn -w 2 -b 0.0.0.0:5000 app:app
"""

import time
import os
from flask import Flask, request, jsonify
from PIL import Image
import io

from model.predict import predict, load_model
from db.database import init_db, log_prediction, get_recent_logs, get_stats

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB max upload

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Startup ───────────────────────────────────────────────────────────────────
with app.app_context():
    init_db()
    try:
        load_model()
        print("✓ Model loaded successfully")
    except FileNotFoundError:
        print("⚠  Model not found — run `python model/train.py` before serving predictions")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """API root — service info and endpoint reference."""
    return jsonify({
        "service":     "Vision Defect Inspection API",
        "version":     "1.0.0",
        "company":     "Swayam Ltd",
        "description": (
            "REST API for automated visual quality inspection. "
            "Classifies product images as PASS (no defect) or FAIL (defect detected) "
            "using a fine-tuned MobileNetV2 TensorFlow model."
        ),
        "endpoints": {
            "POST /predict": "Upload image → PASS/FAIL prediction with confidence",
            "GET  /logs":    "Query recent prediction logs (?limit=50&label=fail)",
            "GET  /stats":   "Aggregate statistics (pass rate, avg confidence, etc.)",
            "GET  /health":  "Service health check",
        },
        "model":  "MobileNetV2 (TensorFlow, transfer learning, CPU)",
        "author": "Bharadwaj Rachuri",
    })


@app.route("/health", methods=["GET"])
def health():
    """Health check — verifies model and database are operational."""
    checks = {}

    # Check model
    try:
        load_model()
        checks["model"] = "ok"
    except Exception as e:
        checks["model"] = f"error: {str(e)}"

    # Check database
    try:
        get_stats()
        checks["database"] = "ok"
    except Exception as e:
        checks["database"] = f"error: {str(e)}"

    status = "healthy" if all(v == "ok" for v in checks.values()) else "degraded"
    http_code = 200 if status == "healthy" else 503

    return jsonify({"status": status, "checks": checks}), http_code


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    POST /predict
    -------------
    Accept a product image and return a PASS/FAIL defect classification.

    Request:
        Content-Type: multipart/form-data
        Field: image  (jpg / png / bmp / webp, max 10 MB)

    Response 200:
        {
            "prediction_id": int,
            "label":         "pass" | "fail",
            "result":        "PASS — No defect detected" | "FAIL — Defect detected",
            "confidence":    float,
            "defect_prob":   float,
            "pass_prob":     float,
            "filename":      str,
            "processing_ms": int
        }

    Response 400: Missing file / invalid format
    Response 500: Inference error
    """
    # ── Validate input ────────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Send as multipart/form-data with key 'image'."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        }), 400

    # ── Inference ─────────────────────────────────────────────────────────────
    try:
        start_ms = time.time()

        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes))

        result = predict(image)

        processing_ms = int((time.time() - start_ms) * 1000)

        # ── Log to SQLite ─────────────────────────────────────────────────────
        prediction_id = log_prediction(
            label=result["label"],
            confidence=result["confidence"],
            defect_prob=result["defect_prob"],
            pass_prob=result["pass_prob"],
            filename=file.filename,
            processing_ms=processing_ms,
        )

        label = result["label"]
        result_text = (
            "PASS — No defect detected"
            if label == "pass"
            else "FAIL — Defect detected"
        )

        return jsonify({
            "prediction_id": prediction_id,
            "label":         label,
            "result":        result_text,
            "confidence":    result["confidence"],
            "defect_prob":   result["defect_prob"],
            "pass_prob":     result["pass_prob"],
            "filename":      file.filename,
            "processing_ms": processing_ms,
        }), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500


@app.route("/logs", methods=["GET"])
def logs_endpoint():
    """
    GET /logs
    ---------
    Retrieve recent prediction logs from the SQLite pipeline.

    Query params:
        limit  int    Max rows to return (default 50, max 200)
        label  str    Filter by 'pass' or 'fail'

    Response 200:
        {
            "count":       int,
            "predictions": [ { id, timestamp, filename, label, confidence, ... } ]
        }
    """
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    label_filter = request.args.get("label", None)
    if label_filter and label_filter not in ("pass", "fail"):
        return jsonify({"error": "label must be 'pass' or 'fail'"}), 400

    logs = get_recent_logs(limit=limit, label_filter=label_filter)
    return jsonify({"count": len(logs), "predictions": logs}), 200


@app.route("/stats", methods=["GET"])
def stats_endpoint():
    """
    GET /stats
    ----------
    Aggregate statistics across all predictions logged in SQLite.

    Response 200:
        {
            "total", "pass_count", "fail_count",
            "pass_rate", "fail_rate",
            "avg_confidence", "avg_defect_prob",
            "avg_processing_ms", "last_prediction_at"
        }
    """
    stats = get_stats()
    return jsonify(stats), 200


# ── Error handlers ────────────────────────────────────────────────────────────

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10 MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found. See GET / for available endpoints."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\nVision Defect Inspection API")
    print(f"Running on http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)
