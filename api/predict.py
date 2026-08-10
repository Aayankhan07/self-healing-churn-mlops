"""
Core inference logic. Called by API routes.
Model and preprocessor loaded once at startup (cached in app.state).
"""

import pandas as pd
import hashlib
import uuid
import json
from datetime import datetime
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


def run_single_prediction(
    request, customer_input, db, healed_actions=None, domain_id: str = "telecom"
) -> PredictionOutput:
    if healed_actions is None:
        healed_actions = []

    state = getattr(getattr(request, "app", request), "state", request)
    from src.domain_registry import (
        load_domain_model,
        load_domain_preprocessor,
        sanitize_domain_id,
    )

    domain_key = sanitize_domain_id(domain_id)

    model = None
    preprocessor = None
    challenger_model = None
    domain_container = None
    model_version = f"{domain_key}-v1"

    if (
        hasattr(state, "model_lock")
        and hasattr(state, "model_registry")
        and domain_key in getattr(state, "model_registry", {})
    ):
        with state.model_lock:
            domain_container = state.model_registry[domain_key]
            model = domain_container.get("model")
            preprocessor = domain_container.get("preprocessor")
            model_version = domain_container.get("version", f"{domain_key}-v1")
            if "challenger" in domain_container and isinstance(
                domain_container["challenger"], dict
            ):
                challenger_model = domain_container["challenger"].get("model")
    elif hasattr(state, "model_lock") and hasattr(state, "model_container"):
        with state.model_lock:
            model = state.model_container["model"]
            preprocessor = state.model_container["preprocessor"]
            model_version = state.model_container["version"]
    else:
        model = getattr(state, "model", None)
        preprocessor = getattr(state, "preprocessor", None)

    if model is None:
        model = load_domain_model(domain_key)
        preprocessor = load_domain_preprocessor(domain_key)

    # Convert to DataFrame
    data = customer_input.model_dump()
    customer_id = data.pop("customerID", None)
    df = pd.DataFrame([data])

    # Engineer features
    df = engineer_features(df)

    # Check for ONNX Runtime acceleration engine
    onnx_engine = domain_container.get("onnx_engine") if domain_container else None

    if onnx_engine is not None:
        try:
            proba = float(onnx_engine.predict_proba(df)[0][1])
        except Exception:
            proba = float(model.predict_proba(df)[0][1])
    else:
        proba = float(model.predict_proba(df)[0][1])

    risk_tier = assign_risk_tier(proba)
    prediction = int(proba >= 0.5)

    # Shadow evaluation if a challenger model is active
    if challenger_model is not None:
        try:
            challenger_proba = float(challenger_model.predict_proba(df)[0][1])
            from api.database import log_shadow_prediction

            log_shadow_prediction(
                db, domain_key, customer_id or "N/A", proba, challenger_proba
            )
        except Exception:
            pass

    # SHAP top factors
    try:
        raw_factors = get_top_shap_factors(model, preprocessor, df, n=3)
        top_factors = [ShapFactor(**f) for f in raw_factors]
    except Exception:
        raw_factors = []
        top_factors = []

    # Upgrade 1: Generate retention playbook
    from src.playbooks import generate_retention_playbook

    recommended_actions = generate_retention_playbook(
        domain_id, raw_factors, data, risk_tier
    )

    # Upgrade 2: Calculate survival timeline
    from src.survival import calculate_survival_curve

    survival = calculate_survival_curve(
        tenure=data.get("tenure", 1), probability=proba, domain_id=domain_id
    )

    prediction_id = str(uuid.uuid4())
    input_hash = hash_input(data)

    # Serialize features and healed actions
    features_json = json.dumps(data)
    healed_actions_json = json.dumps(healed_actions)

    # Log to DB
    log_prediction(
        db,
        prediction_id,
        customer_id,
        input_hash,
        proba,
        risk_tier,
        prediction,
        model_version,
        features_json=features_json,
        healed_actions=healed_actions_json,
        domain_id=domain_id,
    )

    return PredictionOutput(
        customer_id=customer_id,
        churn_probability=round(proba, 4),
        risk_tier=risk_tier,
        prediction=prediction,
        top_factors=top_factors,
        model_version=model_version,
        prediction_id=prediction_id,
        timestamp=datetime.utcnow(),
        healed_actions=healed_actions,
        recommended_actions=recommended_actions,
        time_to_churn_days=survival["time_to_churn_days"],
        risk_horizon_summary=survival["risk_horizon_summary"],
        survival_timeline=survival["survival_timeline"],
    )
