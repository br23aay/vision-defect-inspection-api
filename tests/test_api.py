"""
test_api.py
-----------
pytest test suite for the Vision Defect Inspection API.
Tests all endpoints, database logging, and prediction output structure.

Run:
    pytest tests/ -v
"""

import io
import json
import os
import sys
import pytest
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile

from db.database import init_db, log_prediction, get_recent_logs, get_stats


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_test_image(width=224, height=224, colour=(180, 180, 180)) -> bytes:
    """Create a synthetic JPEG image in memory for testing."""
    img = Image.fromarray(
        np.full((height, width, 3), colour, dtype=np.uint8)
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ── Database tests ────────────────────────────────────────────────────────────

class TestDatabase:
    """Test the SQLite prediction logging pipeline."""

    def setup_method(self):
        """Use a fresh temp file DB before each test."""
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        os.environ["DB_PATH"] = self.tmp.name
        # Reload module to pick up new DB_PATH
        import importlib
        import db.database as db_mod
        importlib.reload(db_mod)
        from db.database import init_db as _init
        _init()

    def teardown_method(self):
        try:
            self.tmp.close()
            os.unlink(self.tmp.name)
        except Exception:
            pass

    def test_log_prediction_returns_id(self):
        row_id = log_prediction(
            label="pass",
            confidence=0.92,
            defect_prob=0.08,
            pass_prob=0.92,
            filename="test.jpg",
            processing_ms=45,
        )
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_get_recent_logs_returns_list(self):
        log_prediction("fail", 0.87, 0.87, 0.13, "fail_01.jpg", 60)
        log_prediction("pass", 0.95, 0.05, 0.95, "pass_01.jpg", 55)
        logs = get_recent_logs(limit=10)
        assert isinstance(logs, list)
        assert len(logs) == 2

    def test_label_filter_pass(self):
        log_prediction("fail", 0.87, 0.87, 0.13, "fail_01.jpg")
        log_prediction("pass", 0.95, 0.05, 0.95, "pass_01.jpg")
        logs = get_recent_logs(label_filter="pass")
        assert all(r["label"] == "pass" for r in logs)

    def test_label_filter_fail(self):
        log_prediction("fail", 0.87, 0.87, 0.13, "fail_01.jpg")
        log_prediction("pass", 0.95, 0.05, 0.95, "pass_01.jpg")
        logs = get_recent_logs(label_filter="fail")
        assert all(r["label"] == "fail" for r in logs)

    def test_get_stats_empty(self):
        stats = get_stats()
        assert stats["total"] == 0
        assert stats["pass_rate"] == 0
        assert stats["fail_rate"] == 0

    def test_get_stats_with_data(self):
        log_prediction("pass", 0.92, 0.08, 0.92)
        log_prediction("pass", 0.88, 0.12, 0.88)
        log_prediction("fail", 0.85, 0.85, 0.15)
        stats = get_stats()
        assert stats["total"] == 3
        assert stats["pass_count"] == 2
        assert stats["fail_count"] == 1
        assert abs(stats["pass_rate"] - 2 / 3) < 0.01

    def test_logs_limit_respected(self):
        for i in range(10):
            log_prediction("pass", 0.9, 0.1, 0.9, f"img_{i}.jpg")
        logs = get_recent_logs(limit=3)
        assert len(logs) == 3

    def test_log_fields_present(self):
        log_prediction("fail", 0.78, 0.78, 0.22, "defect.jpg", 120)
        logs = get_recent_logs(limit=1)
        assert len(logs) == 1
        row = logs[0]
        for field in ["id", "timestamp", "filename", "label", "confidence",
                      "defect_prob", "pass_prob", "processing_ms"]:
            assert field in row, f"Missing field: {field}"


# ── API endpoint tests ────────────────────────────────────────────────────────

@pytest.fixture
def client():
    """Flask test client with in-memory DB."""
    os.environ["DB_PATH"] = ":memory:"
    from app import app
    app.config["TESTING"] = True
    with app.test_client() as client:
        with app.app_context():
            init_db()
        yield client


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        resp = client.get("/")
        assert resp.status_code == 200

    def test_root_contains_service_info(self, client):
        data = json.loads(client.get("/").data)
        assert "service" in data
        assert "endpoints" in data
        assert "POST /predict" in data["endpoints"]


class TestHealthEndpoint:
    def test_health_returns_status(self, client):
        resp = client.get("/health")
        data = json.loads(resp.data)
        assert "status" in data
        assert "checks" in data

    def test_health_checks_database(self, client):
        data = json.loads(client.get("/health").data)
        assert "database" in data["checks"]


class TestLogsEndpoint:
    def test_logs_returns_200(self, client):
        resp = client.get("/logs")
        assert resp.status_code == 200

    def test_logs_empty_initially(self, client):
        data = json.loads(client.get("/logs").data)
        assert "count" in data
        assert "predictions" in data
        assert isinstance(data["predictions"], list)

    def test_logs_invalid_label_returns_400(self, client):
        resp = client.get("/logs?label=unknown")
        assert resp.status_code == 400

    def test_logs_invalid_limit_returns_400(self, client):
        resp = client.get("/logs?limit=abc")
        assert resp.status_code == 400

    def test_logs_limit_param(self, client):
        resp = client.get("/logs?limit=10")
        assert resp.status_code == 200

    def test_logs_label_pass_filter(self, client):
        resp = client.get("/logs?label=pass")
        assert resp.status_code == 200


class TestStatsEndpoint:
    def test_stats_returns_200(self, client):
        resp = client.get("/stats")
        assert resp.status_code == 200

    def test_stats_fields_present(self, client):
        data = json.loads(client.get("/stats").data)
        for field in ["total", "pass_count", "fail_count", "pass_rate", "fail_rate"]:
            assert field in data


class TestPredictEndpoint:
    def test_predict_no_file_returns_400(self, client):
        resp = client.post("/predict")
        assert resp.status_code == 400

    def test_predict_wrong_key_returns_400(self, client):
        img_bytes = make_test_image()
        resp = client.post(
            "/predict",
            data={"file": (io.BytesIO(img_bytes), "test.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_predict_invalid_extension_returns_400(self, client):
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(b"fake data"), "test.txt")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_predict_valid_image_structure(self, client):
        """
        If model is trained, verifies full response structure.
        If model not yet trained, verifies 503 (graceful degradation).
        """
        img_bytes = make_test_image()
        resp = client.post(
            "/predict",
            data={"image": (io.BytesIO(img_bytes), "product.jpg")},
            content_type="multipart/form-data",
        )
        # Either model loaded (200) or not yet trained (503)
        assert resp.status_code in (200, 503)

        if resp.status_code == 200:
            data = json.loads(resp.data)
            assert "label" in data
            assert data["label"] in ("pass", "fail")
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert "defect_prob" in data
            assert "pass_prob" in data
            assert "prediction_id" in data
            assert "processing_ms" in data


class TestErrorHandlers:
    def test_404_returns_json(self, client):
        resp = client.get("/nonexistent")
        assert resp.status_code == 404
        data = json.loads(resp.data)
        assert "error" in data

    def test_405_on_wrong_method(self, client):
        resp = client.get("/predict")
        assert resp.status_code == 405
