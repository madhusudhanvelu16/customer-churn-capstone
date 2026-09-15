# ConnectTel Customer Churn Prediction — AI Application Capstone

[![CI](https://github.com/YOUR_USERNAME/customer-churn-capstone/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/customer-churn-capstone/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

Production-grade AI web service deploying the serialized predictive customer churn pipeline for **ConnectTel Telecom**. Packages a pre-trained LightGBM pipeline into a **FastAPI REST backend** and **Streamlit interactive frontend**, fully containerized with Docker Compose.

> **Docker configuration implemented. Runtime verification complete — see [Docker Verification](#docker-verification) section.**

---

## Table of Contents

- [Business Problem](#business-problem)
- [Architecture](#architecture)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Setup & Local Execution](#setup--local-execution)
- [Testing](#testing)
- [API Endpoints](#api-endpoints)
- [Streamlit Frontend](#streamlit-frontend)
- [Docker Usage](#docker-usage)
- [Environment Variables](#environment-variables)
- [GitHub Actions CI](#github-actions-ci)
- [Cloud Deployment — Render](#cloud-deployment--render)
- [Demo Preparation](#demo-preparation)

---

## Business Problem

Customer acquisition in telecommunications is 5–7× more expensive than retention. This capstone deploys a production ML inference service enabling ConnectTel's customer success teams to **proactively detect churn risk** before customers reach the cancellation decision.

| Dimension | Model Development (Project 1) | Deployment (This Capstone) |
|:---|:---|:---|
| **Location** | `customer-churn-project/` | `customer-churn-capstone/` |
| **Scope** | EDA, training, evaluation, serialization | REST API, frontend, Docker, cloud deployment |
| **Artifact** | Produces `models/churn_pipeline.joblib` | Consumes `model/churn_pipeline.joblib` (read-only) |
| **Validation** | ROC-AUC, recall, confusion matrices | API schema, HTTP status codes, response latencies |

> The deployment service does **not** retrain or fit preprocessing. It loads the exact verified pipeline from Project 1.

---

## Architecture

```
User Browser
     │
     ▼  HTTP / HTTPS
Streamlit Frontend  (port 8501)
     │
     ▼  REST / JSON
FastAPI Backend     (port 8000)
     │
     ├── GET  /          → Service metadata
     ├── GET  /health    → Readiness probe
     ├── GET  /docs      → Interactive Swagger UI
     └── POST /predict   → Pydantic validation + inference
              │
              ▼
     model/churn_pipeline.joblib  (read-only singleton)
              │
              ├── Domain Feature Engineering
              ├── StandardScaler + OneHotEncoder
              └── LightGBM Inference Engine
              │
              ▼
     JSON: prediction + probability + risk_tier
```

**Docker network topology:**

```
                     Docker Network: churn-network
                              │
              ┌───────────────┴───────────────┐
              │                               │
     FastAPI Backend                 Streamlit Frontend
     (churn-backend)                 (churn-frontend)
     Container Port: 8000            Container Port: 8501
     Host: localhost:8000            Host: localhost:8501
              ▲                               │
              └───── HTTP POST /predict ──────┘
              (BACKEND_URL=http://backend:8000)
```

---

## Model Performance

| Metric | Value |
|---|---|
| Model | LightGBM Classifier in scikit-learn Pipeline |
| Training Samples | 5,634 customers |
| Holdout Test Samples | 1,409 customers |
| Cross-Validation ROC-AUC | **0.8469** (±0.0117) |
| Cross-Validation Recall | **0.7920** (±0.0212) |
| Holdout ROC-AUC | **0.8425** |
| Holdout Recall | **80.21%** (300/374 churners detected) |
| Holdout PR-AUC | **0.6605** |
| API + Frontend Tests | **17 / 17 passed** |

---

## Project Structure

```
customer-churn-capstone/
│
├── app/                              # FastAPI backend
│   ├── __init__.py
│   ├── main.py                       # FastAPI app, routes, lifespan
│   ├── model_loader.py               # Singleton model loader
│   ├── predictor.py                  # Inference logic
│   └── schemas.py                    # Pydantic request/response schemas
│
├── frontend/                         # Streamlit frontend
│   ├── api_client.py                 # HTTP client (requests → FastAPI)
│   └── streamlit_app.py              # Interactive UI application
│
├── model/
│   ├── churn_pipeline.joblib         # Serialized LightGBM pipeline (~261 KB)
│   └── model_metadata.json           # Training metadata & metrics
│
├── tests/
│   ├── __init__.py
│   ├── test_api.py                   # 8 FastAPI endpoint tests
│   └── test_frontend.py              # 9 Streamlit API client tests
│
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions CI
│
├── Dockerfile.backend                # FastAPI production image
├── Dockerfile.frontend               # Streamlit production image
├── docker-compose.yml                # Two-service orchestration
├── .dockerignore
├── .env.example                      # Environment variable template
├── .gitignore
├── requirements.txt                  # All dependencies
├── requirements-backend.txt          # Backend-only dependencies
├── requirements-frontend.txt         # Frontend-only dependencies
└── README.md
```

---

## Setup & Local Execution

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Install Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-capstone.git
cd customer-churn-capstone

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Start FastAPI Backend

```bash
# Terminal 1
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- Health: http://localhost:8000/health

### Start Streamlit Frontend

```bash
# Terminal 2
streamlit run frontend/streamlit_app.py --server.port 8501
```

Access at: http://localhost:8501

Custom backend URL:

```bash
# Windows PowerShell
$env:BACKEND_URL="http://localhost:8000"; streamlit run frontend/streamlit_app.py

# Linux / macOS
BACKEND_URL="http://localhost:8000" streamlit run frontend/streamlit_app.py
```

---

## Testing

```bash
pytest tests/ -v
```

**Baseline: 17 / 17 tests passing.**

| Test File | Tests | Coverage |
|---|---|---|
| `test_api.py` | 8 | Root, health, valid predict, schema, probability range, 422 errors, model loader |
| `test_frontend.py` | 9 | API client import, URL config, health success/failure, payload validation, predict success/422, Streamlit compile |

---

## API Endpoints

### Base URL: `http://localhost:8000`

#### `GET /` — Service Metadata
```json
{
  "service": "ConnectTel Customer Churn Prediction API",
  "version": "1.0.0",
  "docs_url": "/docs",
  "health_url": "/health",
  "status": "online"
}
```

#### `GET /health` — Readiness Probe
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "LightGBM Classifier Pipeline",
  "version": "1.0.0"
}
```

#### `POST /predict` — Churn Inference

**Request (19 features required):**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 3,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 79.85,
  "TotalCharges": 239.55
}
```

**Response (High Risk example):**
```json
{
  "prediction": 1,
  "churn_label": "Yes",
  "churn_probability": 0.8731,
  "risk_level": "High Risk"
}
```

**Risk Tier Definitions:**

| Tier | Probability |
|---|---|
| High Risk | ≥ 0.70 |
| Medium Risk | 0.40 – 0.69 |
| Low Risk | < 0.40 |

---

## Streamlit Frontend

The frontend (`frontend/streamlit_app.py`) is a zero-model-duplication SaaS-style interface:

- **19-field customer form** organized into 6 logical sections
- **One-click presets:** High-Risk, Medium-Risk, Low-Risk profiles
- **Live backend health badge** in sidebar
- **Visual results:** probability bar, color-coded risk tier banner
- **Retention guidance:** actionable recommendations based on risk drivers

The frontend communicates exclusively via HTTP with the FastAPI backend. It does **not** load `churn_pipeline.joblib`.

---

## Docker Usage

### Prerequisites
- Docker Engine 24+ and Docker Compose v2+

```bash
# Build images
docker compose build

# Start in detached mode
docker compose up -d

# Check status
docker compose ps
docker compose logs -f

# Verify endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs
# Open http://localhost:8501 in browser

# Stop
docker compose down
```

### Docker Verification

Docker configuration implemented with two-service architecture:

- **Backend image:** `python:3.11-slim` + `libgomp1` (LightGBM OpenMP) + `curl` (healthcheck)
- **Frontend image:** `python:3.11-slim` + `curl` (healthcheck)
- **Network:** `churn-network` (bridge) — frontend calls `http://backend:8000`, not `localhost`
- **Healthchecks:** Backend checks `/health` every 20s; frontend depends on `service_healthy`

> Runtime verification performed — see Phase 4 completion report for results.

---

## Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

| Variable | Local Default | Docker Default | Description |
|---|---|---|---|
| `BACKEND_URL` | `http://localhost:8000` | `http://backend:8000` | Frontend → backend URL |
| `MODEL_PATH` | `model/churn_pipeline.joblib` | `model/churn_pipeline.joblib` | Model artifact path |
| `API_HOST` | `0.0.0.0` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | `8000` | FastAPI bind port |
| `LOG_LEVEL` | `info` | `info` | Logging verbosity |
| `PYTHONUNBUFFERED` | `1` | `1` | Real-time stdout logs |

---

## GitHub Actions CI

Automated CI runs on every `push` and `pull_request`:

```yaml
# .github/workflows/ci.yml
# 1. Checkout
# 2. Set up Python 3.11
# 3. Install libgomp1 (LightGBM OpenMP support)
# 4. Install pip dependencies
# 5. Compile all source files
# 6. pytest tests/ -v  (17 tests)
```

---

## Cloud Deployment — Render

**Selected platform: Render** — best fit for this student capstone.

| Criterion | Render |
|---|---|
| Docker support | ✅ Native |
| Environment variables | ✅ Dashboard + secrets |
| Separate backend/frontend services | ✅ Two independent web services |
| Free tier | ✅ Suitable for demos |
| Auto-deploy from GitHub | ✅ On push to main |
| Public HTTPS URLs | ✅ Automatic TLS |

### Deployment Steps

**1. Create GitHub repositories and push both projects** (see [GitHub Push Commands](#github-push-commands) below).

**2. Deploy FastAPI Backend on Render:**

- New Web Service → Connect `customer-churn-capstone` repo
- Runtime: **Docker**
- Dockerfile path: `Dockerfile.backend`
- Start command: *(auto-detected from CMD in Dockerfile)*
- Environment variables:
  ```
  MODEL_PATH=model/churn_pipeline.joblib
  PYTHONUNBUFFERED=1
  ```
- Port: Render injects `PORT` — see `render.yaml` for configuration

**3. Deploy Streamlit Frontend on Render:**

- New Web Service → Connect `customer-churn-capstone` repo
- Runtime: **Docker**
- Dockerfile path: `Dockerfile.frontend`
- Environment variables:
  ```
  BACKEND_URL=https://your-backend-service.onrender.com
  PYTHONUNBUFFERED=1
  ```

See `render.yaml` in the repository root for the full Infrastructure-as-Code configuration.

> **Deployment status:** Configuration prepared. Actual deployment and public URL verification pending GitHub push.

---

## Demo Preparation

To demonstrate the full capstone:

1. **GitHub Repository** — Show project structure on GitHub
2. **GitHub Actions** — Show CI passing (all 17 tests green)
3. **FastAPI Docs** — `GET http://localhost:8000/docs`
4. **Health Check** — `GET http://localhost:8000/health`
5. **High-Risk POST** — Use the High Risk preset in Streamlit or send directly:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"gender":"Female","SeniorCitizen":0,"Partner":"No","Dependents":"No","tenure":3,"PhoneService":"Yes","MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No","DeviceProtection":"No","TechSupport":"No","StreamingTV":"Yes","StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes","PaymentMethod":"Electronic check","MonthlyCharges":79.85,"TotalCharges":239.55}'
   ```
   Expected: `~87.31% — High Risk`

6. **Low-Risk POST** — Use the Low Risk preset:
   Expected: `~4.01% — Low Risk`

7. **Streamlit Frontend** — `http://localhost:8501`
8. **Docker** — `docker compose up -d && docker compose ps`
9. **Public URL** — Available after Render deployment

---

## GitHub Push Commands

After creating your GitHub repositories, run:

```bash
# customer-churn-capstone
cd customer-churn-capstone
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-capstone.git
git push -u origin main

# customer-churn-project
cd ../customer-churn-project
git remote add origin https://github.com/YOUR_USERNAME/customer-churn-project.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

---

*Capstone Deployment Phase — ConnectTel Churn Prediction System*
