"""
Core inference logic. Called by API routes.
Model and preprocessor loaded once at startup (cached in app.state).
"""
import pandas as pd
import numpy as np
import hashlib
import uuid
from datetime import datetime
from fastapi import Request
from src.features import engineer_features
from src.evaluate import get_top_shap_factors
from api.schemas import PredictionOutput, ShapFactor
from api.database import log_prediction


RISK_THRESHOLDS = {"low": 0.35, "high": 0.65}


def assign_risk_tier(probability: float) -> str:
    if probability >= RISK_THRESHOLDS["high"]:
        return "High"
    elif probability >= RISK_THRESHOLDS["low"]:
        return "Medium"
    return "Low"


def hash_input(data: dict) -> str:
    """SHA256 of input dict — stored in DB instead of raw PII."""
    return hashlib.sha256(str(sorted(data.items())).encode()).hexdigest()[:16]


def run_single_prediction(request, customer_input, db) -> PredictionOutput:
    # Use request.app.state or fallback to local files if model is not loaded in app.state
    model = getattr(getattr(request, "app", request), "state", request).model
    preprocessor = getattr(getattr(request, "app", request), "state", request).preprocessor
    model_version = getattr(getattr(request, "app", request), "state", request).model_version

    # Convert to DataFrame
    data = customer_input.model_dump()
    customer_id = data.pop("customerID", None)
    df = pd.DataFrame([data])

    # Engineer features
    df = engineer_features(df)

    # Predict
    proba = float(model.predict_proba(df)[0][1])
    risk_tier = assign_risk_tier(proba)
    prediction = int(proba >= 0.5)

    # SHAP top factors
    try:
        raw_factors = get_top_shap_factors(model, preprocessor, df, n=3)
        top_factors = [ShapFactor(**f) for f in raw_factors]
    except Exception:
        top_factors = []

    prediction_id = str(uuid.uuid4())
    input_hash = hash_input(data)

    # Log to DB
    log_prediction(db, prediction_id, customer_id, input_hash,
                   proba, risk_tier, prediction, model_version)

    return PredictionOutput(
        customer_id=customer_id,
        churn_probability=round(proba, 4),
        risk_tier=risk_tier,
        prediction=prediction,
        top_factors=top_factors,
        model_version=model_version,
        prediction_id=prediction_id,
        timestamp=datetime.utcnow(),
    )
