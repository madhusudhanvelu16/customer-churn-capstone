"""Pydantic schemas for ConnectTel Customer Churn API."""

from typing import Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class CustomerInputSchema(BaseModel):
    """Customer attributes schema strictly derived from the Telco dataset.
    
    Contains all 19 raw feature inputs expected by the serialized pipeline.
    """
    gender: Literal["Female", "Male"] = Field(
        ..., description="Gender of the customer (Female or Male)"
    )
    SeniorCitizen: int = Field(
        ..., ge=0, le=1, description="Whether customer is a senior citizen (1) or not (0)"
    )
    Partner: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has a partner (Yes or No)"
    )
    Dependents: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has dependents (Yes or No)"
    )
    tenure: int = Field(
        ..., ge=0, le=120, description="Number of months customer has stayed with company"
    )
    PhoneService: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has a phone service (Yes or No)"
    )
    MultipleLines: Literal["No", "Yes", "No phone service"] = Field(
        ..., description="Whether customer has multiple phone lines"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Type of customer's internet service provider"
    )
    OnlineSecurity: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has online security add-on"
    )
    OnlineBackup: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has online backup add-on"
    )
    DeviceProtection: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has device protection add-on"
    )
    TechSupport: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has tech support add-on"
    )
    StreamingTV: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has streaming TV service"
    )
    StreamingMovies: Literal["No", "Yes", "No internet service"] = Field(
        ..., description="Whether customer has streaming movies service"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="The contract term of the customer"
    )
    PaperlessBilling: Literal["Yes", "No"] = Field(
        ..., description="Whether customer has paperless billing enabled"
    )
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., description="Customer's payment method")
    MonthlyCharges: float = Field(
        ..., ge=0.0, le=1000.0, description="Amount charged to customer monthly"
    )
    TotalCharges: float = Field(
        ..., ge=0.0, le=100000.0, description="Total amount charged to customer"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
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
        }
    )


class PredictionResponseSchema(BaseModel):
    """Structured inference response schema."""
    prediction: int = Field(..., description="Binary classification (0: Retained, 1: Churn)")
    churn_label: Literal["Yes", "No"] = Field(
        ..., description="Human-readable churn label ('Yes' for churn, 'No' for retained)"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Estimated posterior probability of churn"
    )
    risk_level: Literal["Low Risk", "Medium Risk", "High Risk"] = Field(
        ..., description="Risk tier based on threshold segmentation"
    )


class HealthResponseSchema(BaseModel):
    """Health check status response."""
    status: str = Field(..., description="Service health state")
    model_loaded: bool = Field(..., description="Whether ML pipeline is active in memory")
    model_name: Optional[str] = Field(None, description="Identifier of production model")
    version: str = Field(..., description="API Version")
