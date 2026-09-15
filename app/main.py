"""FastAPI application for ConnectTel Customer Churn Prediction Service."""

import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model_loader import load_model, get_loaded_model
from app.predictor import predict_churn
from app.schemas import CustomerInputSchema, PredictionResponseSchema, HealthResponseSchema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("churn_api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context: initialize model on startup."""
    logger.info("Initializing ConnectTel Churn API Service...")
    try:
        model = load_model()
        logger.info("Model pipeline loaded into memory successfully.")
    except Exception as exc:
        logger.critical(f"FATAL: Failed to initialize model pipeline on startup: {exc}")
    yield
    logger.info("Shutting down ConnectTel Churn API Service...")


app = FastAPI(
    title="ConnectTel Customer Churn Prediction API",
    description=(
        "Production-grade REST API serving ConnectTel's serialized machine learning pipeline "
        "to predict customer churn probability and risk tier."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """Welcome endpoint with API metadata and documentation links."""
    return {
        "service": "ConnectTel Customer Churn Prediction API",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health",
        "status": "online",
    }


@app.get(
    "/health",
    response_model=HealthResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["System"],
)
async def health_check():
    """Health check endpoint to verify service and model readiness.
    
    Returns HTTP 200 when ready.
    """
    try:
        model = get_loaded_model()
        is_loaded = model is not None
        model_name = "LightGBM Classifier Pipeline" if is_loaded else "Unloaded"
    except Exception as exc:
        logger.error(f"Health check model verification failed: {exc}")
        is_loaded = False
        model_name = None

    if not is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not ready or failed to load.",
        )

    return HealthResponseSchema(
        status="healthy",
        model_loaded=True,
        model_name=model_name,
        version="1.0.0",
    )


@app.post(
    "/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    tags=["Inference"],
)
async def predict(payload: CustomerInputSchema) -> PredictionResponseSchema:
    """Predict customer churn status and probability.
    
    - Validates incoming customer attributes.
    - Executes end-to-end serialized pipeline inference.
    - Returns binary prediction, churn probability, and assigned risk tier.
    """
    try:
        result = predict_churn(payload)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Unexpected inference error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing the prediction request. Please verify inputs.",
        )
