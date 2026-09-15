"""Model loader module for ConnectTel Customer Churn inference pipeline.

Responsibilities:
- Load churn_pipeline.joblib exactly once.
- Fail clearly if the artifact is missing or corrupted.
- Provide reusable model singleton without retraining or refitting.
"""

import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Optional
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger("churn_api.model_loader")


# Exact DomainFeatureEngineer definition required by serialized pipeline unpickler
class DomainFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer to engineer telecom domain features.
    
    Preserves exact logic from Project 1 model training.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        tc = pd.to_numeric(X_out.get("TotalCharges", 0), errors="coerce").fillna(0.0)
        mc = pd.to_numeric(X_out.get("MonthlyCharges", 0), errors="coerce").fillna(0.0)
        X_out["charge_ratio"] = mc / (tc + 1.0)

        sec_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
        sec_count = np.zeros(len(X_out))
        for col in sec_cols:
            if col in X_out.columns:
                sec_count += (X_out[col] == "Yes").astype(int)
        X_out["security_services_count"] = sec_count

        st_cols = ["StreamingTV", "StreamingMovies"]
        st_count = np.zeros(len(X_out))
        for col in st_cols:
            if col in X_out.columns:
                st_count += (X_out[col] == "Yes").astype(int)
        X_out["streaming_services_count"] = st_count

        has_fam = np.zeros(len(X_out))
        if "Partner" in X_out.columns:
            has_fam = has_fam | (X_out["Partner"] == "Yes")
        if "Dependents" in X_out.columns:
            has_fam = has_fam | (X_out["Dependents"] == "Yes")
        X_out["has_family"] = has_fam.astype(int)

        return X_out


# Register alias in sys.modules so joblib unpickling finds DomainFeatureEngineer
if "src.feature_engineering" not in sys.modules:
    src_mod = types.ModuleType("src")
    fe_mod = types.ModuleType("src.feature_engineering")
    fe_mod.DomainFeatureEngineer = DomainFeatureEngineer
    sys.modules["src"] = src_mod
    sys.modules["src.feature_engineering"] = fe_mod


_MODEL_INSTANCE: Optional[Any] = None


def resolve_model_path() -> Path:
    """Determine the file path to the production model artifact."""
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        p = Path(env_path)
        if p.is_absolute():
            return p
        # Relative to project root
        return (Path(__file__).resolve().parent.parent / p).resolve()

    # Default location: customer-churn-capstone/model/churn_pipeline.joblib
    default_path = (Path(__file__).resolve().parent.parent / "model" / "churn_pipeline.joblib").resolve()
    return default_path


def load_model(model_path: Optional[Path] = None, force_reload: bool = False) -> Any:
    """Load serialized model pipeline artifact.
    
    Args:
        model_path: Optional custom Path to model file.
        force_reload: If True, forces reload even if already in memory.
        
    Returns:
        Fitted scikit-learn Pipeline object ready for inference.
        
    Raises:
        FileNotFoundError: If the model file does not exist.
        RuntimeError: If deserialization fails.
    """
    global _MODEL_INSTANCE

    if _MODEL_INSTANCE is not None and not force_reload:
        return _MODEL_INSTANCE

    target_path = model_path if model_path is not None else resolve_model_path()

    if not target_path.exists():
        msg = (
            f"Production model artifact not found at: {target_path}. "
            "Please ensure 'model/churn_pipeline.joblib' is present."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info(f"Loading production model pipeline from {target_path}...")
    try:
        loaded_pipeline = joblib.load(target_path)
    except Exception as exc:
        msg = f"Failed to deserialize model pipeline from {target_path}: {exc}"
        logger.error(msg)
        raise RuntimeError(msg) from exc

    _MODEL_INSTANCE = loaded_pipeline
    logger.info("Production model pipeline successfully loaded and cached in memory.")
    return _MODEL_INSTANCE


def get_loaded_model() -> Any:
    """Retrieve currently loaded model or trigger initialization."""
    if _MODEL_INSTANCE is None:
        return load_model()
    return _MODEL_INSTANCE
