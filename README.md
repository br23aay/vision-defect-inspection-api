# Vision-Based Defect & Quality Inspection API

**Swayam Ltd — ML Engineering | Production Service**

A production-ready REST API for automated visual quality inspection. Classifies product images as **PASS** (no defect) or **FAIL** (defect detected) using a fine-tuned MobileNetV2 TensorFlow model served via Flask, with SQLite prediction logging, CI/CD via GitHub Actions, and AWS deployment.

---

## Stack

| Component | Technology |
|---|---|
| Vision Model | TensorFlow 2.17 · MobileNetV2 · Transfer Learning |
| API Framework | Flask 3.1 |
| Database Pipeline | SQLite · Custom SQL logging & analytics |
| CI/CD | GitHub Actions (lint → test → build → deploy) |
| Deployment | Docker · AWS Elastic Beanstalk / ECR |
| Testing | pytest · 25+ tests |

---

## Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Upload image → PASS/FAIL prediction with confidence |
| `GET` | `/logs` | Query recent prediction logs (`?limit=50&label=fail`) |
| `GET` | `/stats` | Aggregate statistics (pass rate, avg confidence, etc.) |
| `GET` | `/health` | Service health check (model + DB status) |
| `GET` | `/` | API info and endpoint reference |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/br23aay/vision-defect-inspection-api
cd vision-defect-inspection-api
pip install -r requirements.txt

# 2. Generate synthetic training data
python data/generate_synthetic.py

# 3. Train the TensorFlow model (CPU, ~10–20 min)
python model/train.py

# 4. Run the Flask API
python app.py

# 5. Run tests
pytest tests/ -v
```

---

## API Usage

### Predict — PASS/FAIL

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/product.jpg"
```

```json
{
  "prediction_id": 1,
  "label": "fail",
  "result": "FAIL — Defect detected",
  "confidence": 0.923,
  "defect_prob": 0.923,
  "pass_prob": 0.077,
  "filename": "product.jpg",
  "processing_ms": 47
}
```

### Query Logs

```bash
# Last 20 failed predictions
curl "http://localhost:5000/logs?limit=20&label=fail"
```

### Aggregate Statistics

```bash
curl http://localhost:5000/stats
```

```json
{
  "total": 150,
  "pass_count": 112,
  "fail_count": 38,
  "pass_rate": 0.7467,
  "fail_rate": 0.2533,
  "avg_confidence": 0.891,
  "avg_defect_prob": 0.312,
  "avg_processing_ms": 52.3,
  "last_prediction_at": "2026-06-15T14:23:11.042"
}
```

---

## Model Architecture

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet weights, frozen base)
    ↓
GlobalAveragePooling2D
    ↓
Dropout(0.3)
    ↓
Dense(128, relu)
    ↓
Dropout(0.2)
    ↓
Dense(1, sigmoid) → P(pass)
```

**Training:**
- Phase 1: Head only, 5 epochs, LR=1e-3
- Phase 2: Fine-tune last 20 MobileNetV2 layers, 3 epochs, LR=1e-4
- Augmentation: rotation, flip, zoom, shift

---

## CI/CD Pipeline

```
Push to main / PR
    ↓
[1] flake8 lint
    ↓
[2] pytest (25+ tests, in-memory DB)
    ↓
[3] Docker build + smoke test
    ↓
[4] Push to AWS ECR → Deploy to Elastic Beanstalk
    (main branch only)
```

---

## Skills Evidenced

- **TensorFlow** — MobileNetV2 transfer learning, custom classification head, fine-tuning
- **Vision Models** — Image classification pipeline, data augmentation, preprocessing
- **Flask / REST API** — 5 production endpoints, error handling, file upload validation
- **SQL Pipelines** — SQLite prediction logging, indexed queries, aggregate analytics
- **CI/CD** — GitHub Actions: lint → test → build → deplhoy (4-stage pipeline)
- **AWS** — ECR image registry, Elastic Beanstalk deployment, Docker containerisation

---

## Author

**Bharadwaj Rachuri** — ML & AI Engineer  
[br23aay.github.io](https://br23aay.github.io) · [github.com/br23aay](https://github.com/br23aay)  
done by bharadwaj rachuri
