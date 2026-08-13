"""
Scoring routes: single prediction, batch prediction, and CSV upload.
"""

import logging
import tempfile
import threading
import time
from io import StringIO

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.database import get_db, log_self_healing_event
from api.dependencies import verify_api_key
from api.predict import run_single_prediction
from api.schemas import (
    BatchOutput,
    CustomerInput,
    LaxBatchInput,
    PredictionOutput,
    build_request_model,
    unknown_fields,
)
from api.services import model_registry
from api.services.drift_service import run_drift_check
from src.domains import get_domain_spec
from src.healing import heal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict", response_model=PredictionOutput, dependencies=[Depends(verify_api_key)]
)
def predict(
    request: Request,
    customer: dict,
    domain: str = "telecom",
    db: Session = Depends(get_db),
):
    app = request.app
    domain_val = customer.pop("domain", domain)
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    spec = get_domain_spec(domain_val)

    # Ingestion Self-Healing, against this domain's schema
    healed_dict, healed_actions = heal(customer, spec)

    # Strict validation
    try:
        validated_customer = build_request_model(spec)(**healed_dict)
    except Exception as e:
        raise HTTPException(
            status_code=422, detail=f"Validation failed after healing: {str(e)}"
        )

    customer_id = healed_dict.get("customerID")
    if healed_actions:
        log_self_healing_event(
            db,
            "data_quality",
            f"Healed prediction input for Customer {customer_id or 'N/A'}. Actions: {', '.join(healed_actions)}",
            domain_id=spec.key,
        )

    unexpected = unknown_fields(customer, spec)
    if unexpected:
        log_self_healing_event(
            db,
            "data_quality",
            f"Prediction input for Customer {customer_id or 'N/A'} carried fields "
            f"not declared by domain '{spec.key}': {', '.join(unexpected)}. "
            f"They were accepted but not used for scoring.",
            domain_id=spec.key,
        )

    result = run_single_prediction(
        app, validated_customer, db, healed_actions=healed_actions, domain_id=domain_val
    )
    if model_registry.count_prediction(app):
        threading.Thread(target=run_drift_check, args=(app, domain_val)).start()
    return result


@router.post(
    "/predict/batch", response_model=BatchOutput, dependencies=[Depends(verify_api_key)]
)
def predict_batch(
    request: Request,
    batch: LaxBatchInput,
    domain: str = "telecom",
    db: Session = Depends(get_db),
):
    app = request.app
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.time()

    spec = get_domain_spec(domain)
    request_model = build_request_model(spec)

    predictions = []
    for customer in batch.customers:
        healed_dict, healed_actions = heal(customer, spec)
        try:
            validated_customer = request_model(**healed_dict)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Validation failed for a batch item after healing: {str(e)}",
            )

        customer_id = healed_dict.get("customerID")
        if healed_actions:
            log_self_healing_event(
                db,
                "data_quality",
                f"Healed batch input for Customer {customer_id or 'N/A'}. Actions: {', '.join(healed_actions)}",
                domain_id=spec.key,
            )

        pred = run_single_prediction(
            app, validated_customer, db, healed_actions=healed_actions, domain_id=domain
        )
        predictions.append(pred)

    high_risk = sum(1 for p in predictions if p.risk_tier == "High")

    # A batch that straddles the drift-check boundary triggers exactly one check.
    if model_registry.count_prediction(app, len(predictions)):
        threading.Thread(target=run_drift_check, args=(app, domain)).start()

    return BatchOutput(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
        processing_time_ms=round((time.time() - t0) * 1000, 1),
    )


@router.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_csv(
    request: Request,
    file: UploadFile = File(...),
    domain: str = "telecom",
    db: Session = Depends(get_db),
):
    app = request.app
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="File must be a CSV")
    content = await file.read()
    df = pd.read_csv(StringIO(content.decode("utf-8")))

    required = list(CustomerInput.model_fields.keys())
    required.remove("customerID")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"missing_columns": missing})

    results = []
    for _, row in df.iterrows():
        data = row.to_dict()
        customer = CustomerInput(**data)
        pred = run_single_prediction(app, customer, db, domain_id=domain)
        results.append(
            {
                "customerID": pred.customer_id,
                "churn_probability": pred.churn_probability,
                "risk_tier": pred.risk_tier,
                "top_factor": (
                    pred.top_factors[0].feature if pred.top_factors else "N/A"
                ),
            }
        )

    result_df = df.copy()
    result_df["churn_probability"] = [r["churn_probability"] for r in results]
    result_df["risk_tier"] = [r["risk_tier"] for r in results]
    result_df["top_factor"] = [r["top_factor"] for r in results]

    # Per-request temp file — a shared fixed path lets concurrent uploads
    # overwrite each other's results before the response is streamed.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline="", encoding="utf-8"
    ) as tmp:
        out_path = tmp.name
        result_df.to_csv(tmp, index=False)
    return FileResponse(out_path, media_type="text/csv", filename="predictions.csv")
