"""Inference and predictor service for ConnectTel Customer Churn."""

import logging
from typing import Dict, Any, Union
import pandas as pd

from app.model_loader import get_loaded_model
from app.schemas import CustomerInputSchema, PredictionResponseSchema

logger = logging.getLogger("churn_api.predictor")

# Exact feature ordering expected by the trained pipeline
PIPELINE_FEATURE_ORDER = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def determine_risk_level(probability: float) -> str:
    """Categorize churn probability according to Project 1 risk boundaries.
    
    - High Risk: >= 0.70
    - Medium Risk: 0.40 <= prob < 0.70
    - Low Risk: < 0.40
    """
    if probability >= 0.70:
        return "High Risk"
    elif probability >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"


def predict_churn(
    customer_input: Union[CustomerInputSchema, Dict[str, Any]],
) -> PredictionResponseSchema:
    """Run end-to-end churn prediction through the serialized pipeline.
    
    Args:
        customer_input: Validated customer data (schema or dict).
        
    Returns:
        PredictionResponseSchema with prediction, probability, label, and risk tier.
    """
    # 1. Retrieve cached model
    pipeline = get_loaded_model()

    # 2. Convert input into DataFrame with explicit feature ordering
    if isinstance(customer_input, CustomerInputSchema):
        data_dict = customer_input.model_dump()
    else:
        data_dict = customer_input

    # Ensure all required features are present
    df_payload = pd.DataFrame([{col: data_dict[col] for col in PIPELINE_FEATURE_ORDER}])

    # 3. Model inference (pipeline handles internal feature engineering & scaling)
    logger.debug(f"Passing payload to pipeline: {df_payload.to_dict(orient='records')}")

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(df_payload)
        churn_prob = float(probs[0, 1])
    else:
        churn_prob = float(pipeline.predict(df_payload)[0])

    # Default decision boundary from model training
    binary_pred = int(churn_prob >= 0.50)
    churn_label = "Yes" if binary_pred == 1 else "No"
    risk_level = determine_risk_level(churn_prob)

    response = PredictionResponseSchema(
        prediction=binary_pred,
        churn_label=churn_label,
        churn_probability=round(churn_prob, 4),
        risk_level=risk_level,
    )

    logger.info(
        f"Inference completed: pred={binary_pred}, prob={response.churn_probability}, risk={risk_level}"
    )
    return response
