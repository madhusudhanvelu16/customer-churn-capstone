"""HTTP API client for communicating with the ConnectTel FastAPI backend."""

import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("churn_frontend.api_client")

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_HEALTH = 3.0
DEFAULT_TIMEOUT_PREDICT = 6.0


def get_backend_url() -> str:
    """Retrieve backend API base URL from environment or fallback default."""
    url = os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL).strip()
    return url.rstrip("/")


def check_health(backend_url: Optional[str] = None) -> Dict[str, Any]:
    """Query backend /health endpoint to check service and model readiness.
    
    Args:
        backend_url: Optional override for backend base URL.
        
    Returns:
        Dictionary with status, model_loaded, model_name, and connection state.
    """
    base_url = backend_url.rstrip("/") if backend_url else get_backend_url()
    health_url = f"{base_url}/health"

    try:
        response = requests.get(health_url, timeout=DEFAULT_TIMEOUT_HEALTH)
        if response.status_code == 200:
            data = response.json()
            return {
                "is_available": True,
                "status": data.get("status", "healthy"),
                "model_loaded": data.get("model_loaded", False),
                "model_name": data.get("model_name", "Inference Pipeline"),
                "version": data.get("version", "1.0.0"),
                "error_message": None,
            }
        else:
            return {
                "is_available": False,
                "status": "unhealthy",
                "model_loaded": False,
                "model_name": None,
                "version": None,
                "error_message": f"Backend returned HTTP {response.status_code}: {response.text}",
            }
    except requests.exceptions.ConnectionError:
        return {
            "is_available": False,
            "status": "offline",
            "model_loaded": False,
            "model_name": None,
            "version": None,
            "error_message": f"Connection refused at {base_url}. Ensure the FastAPI server is running.",
        }
    except requests.exceptions.Timeout:
        return {
            "is_available": False,
            "status": "timeout",
            "model_loaded": False,
            "model_name": None,
            "version": None,
            "error_message": f"Health check timed out after {DEFAULT_TIMEOUT_HEALTH}s.",
        }
    except Exception as exc:
        return {
            "is_available": False,
            "status": "error",
            "model_loaded": False,
            "model_name": None,
            "version": None,
            "error_message": f"Unexpected connection error: {str(exc)}",
        }


def predict_churn(
    payload: Dict[str, Any], backend_url: Optional[str] = None
) -> Dict[str, Any]:
    """Send customer profile to backend POST /predict for inference.
    
    Args:
        payload: Dictionary containing all 19 required raw features.
        backend_url: Optional override for backend base URL.
        
    Returns:
        Dictionary with success status, prediction data, and error message if failed.
    """
    base_url = backend_url.rstrip("/") if backend_url else get_backend_url()
    predict_url = f"{base_url}/predict"

    try:
        response = requests.post(
            predict_url, json=payload, timeout=DEFAULT_TIMEOUT_PREDICT
        )

        if response.status_code == 200:
            result = response.json()
            return {
                "success": True,
                "data": result,
                "status_code": 200,
                "error_message": None,
            }
        elif response.status_code == 422:
            detail = response.json().get("detail", "Validation error")
            return {
                "success": False,
                "data": None,
                "status_code": 422,
                "error_message": f"Invalid customer input: {detail}",
            }
        else:
            return {
                "success": False,
                "data": None,
                "status_code": response.status_code,
                "error_message": f"Backend error (HTTP {response.status_code}): {response.text}",
            }
    except requests.exceptions.ConnectionError:
        return {
            "success": False,
            "data": None,
            "status_code": None,
            "error_message": "Could not connect to FastAPI backend. Please check server status.",
        }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "data": None,
            "status_code": None,
            "error_message": f"Inference request timed out after {DEFAULT_TIMEOUT_PREDICT}s.",
        }
    except Exception as exc:
        return {
            "success": False,
            "data": None,
            "status_code": None,
            "error_message": f"An error occurred during prediction request: {str(exc)}",
        }
