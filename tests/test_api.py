"""Test suite for ConnectTel Customer Churn FastAPI service."""

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

# Ensure capstone root is on sys.path
CAPSTONE_ROOT = Path(__file__).resolve().parent.parent
if str(CAPSTONE_ROOT) not in sys.path:
    sys.path.insert(0, str(CAPSTONE_ROOT))

from app.main import app
from app.model_loader import load_model, resolve_model_path


@pytest.fixture(scope="module")
def client():
    """Create a test client inside FastAPI lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def valid_customer_payload():
    """Actual customer record from Project 1 dataset (Index 0)."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


@pytest.fixture
def high_risk_customer_payload():
    """High churn risk customer record."""
    return {
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
        "TotalCharges": 239.55,
    }


def test_root_endpoint(client):
    """Verify welcome root endpoint returns HTTP 200 and links."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "docs_url" in data


def test_health_endpoint(client):
    """1. Test /health returns 200 and confirms model readiness."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_valid_predict_request(client, valid_customer_payload):
    """2. Test valid /predict request executes and returns 200."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "churn_probability" in data
    assert "churn_label" in data
    assert "risk_level" in data


def test_predict_response_schema(client, high_risk_customer_payload):
    """6. Test prediction response strictly adheres to expected schema."""
    response = client.post("/predict", json=high_risk_customer_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["prediction"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert isinstance(data["churn_probability"], float)
    assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]

    # Match binary prediction with label
    if data["prediction"] == 1:
        assert data["churn_label"] == "Yes"
    else:
        assert data["churn_label"] == "No"


def test_probability_range(client, valid_customer_payload):
    """7. Test churn probability is bounded between 0.0 and 1.0."""
    response = client.post("/predict", json=valid_customer_payload)
    assert response.status_code == 200
    prob = response.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_invalid_categorical_input(client, valid_customer_payload):
    """3. Test invalid categorical value triggers validation error 422."""
    payload = valid_customer_payload.copy()
    payload["Contract"] = "Five year"  # Invalid contract duration
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_required_field(client, valid_customer_payload):
    """4. Test missing field triggers validation error 422."""
    payload = valid_customer_payload.copy()
    del payload["MonthlyCharges"]  # Missing required field
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_data_type(client, valid_customer_payload):
    """5. Test invalid data type triggers validation error 422."""
    payload = valid_customer_payload.copy()
    payload["tenure"] = "thirty-five"  # String instead of int
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_loading_failure_handling():
    """8. Test model loader raises FileNotFoundError when artifact is missing."""
    non_existent_path = Path("non_existent_model.joblib")
    with pytest.raises(FileNotFoundError):
        load_model(model_path=non_existent_path, force_reload=True)
