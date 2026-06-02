"""
FastAPI application entrypoint.
All routes defined here. Model loaded once on startup.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import time
import os
import mlflow.sklearn
import joblib
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from typing import Optional
from io import StringIO
import tempfile

from api.schemas import (
    CustomerInput, PredictionOutput, BatchInput, BatchOutput,
    HealthResponse, MetricsResponse, DriftStatusResponse
)
from api.predict import run_single_prediction
from api.database import (
    init_db, get_db, count_predictions, predictions_last_n_days,
    risk_distribution, last_n_inputs, get_latest_drift, log_drift_report
)
from src.features import engineer_features
from api.drift import generate_drift_report

START_TIME = time.time()
API_KEY = os.getenv("API_KEY", "dev-key-change-in-prod")

app = FastAPI(
    title="ChurnGuard API",
    description="Customer churn prediction — predict, explain, monitor.",
    version="1.0.0",
)


def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.on_event("startup")
async def startup():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    model_name = os.getenv("MLFLOW_MODEL_NAME", "churn_model")
    model_stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")

    try:
        app.state.model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")
        app.state.model_version = model_stage
        app.state.model_loaded = True
    except Exception as e:
        # Fallback to local model if MLflow server is offline
        local_model_path = "models/model.joblib"
        if os.path.exists(local_model_path):
            try:
                app.state.model = joblib.load(local_model_path)
                app.state.model_version = "local-fallback"
                app.state.model_loaded = True
            except Exception:
                app.state.model_loaded = False
                app.state.model = None
                app.state.model_version = "unknown"
        else:
            app.state.model_loaded = False
            app.state.model = None
            app.state.model_version = "unknown"

    app.state.preprocessor = joblib.load("models/preprocessor.joblib")
    app.state.reference_data = pd.read_csv("data/processed/train.csv").drop(
        columns=["Churn"], errors="ignore"
    )
    app.state.prediction_counter = 0
    app.state.drift_check_every = int(os.getenv("DRIFT_CHECK_EVERY_N", 100))
    init_db()


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if app.state.model_loaded else "degraded",
        model_loaded=app.state.model_loaded,
        model_version=app.state.model_version,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.post("/predict", response_model=PredictionOutput, dependencies=[Depends(verify_api_key)])
def predict(customer: CustomerInput, db: Session = Depends(get_db)):
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    result = run_single_prediction(app, customer, db)  # pass app for state access
    app.state.prediction_counter += 1
    if app.state.prediction_counter % app.state.drift_check_every == 0:
        _run_drift_check(db)
    return result


@app.post("/predict/batch", response_model=BatchOutput, dependencies=[Depends(verify_api_key)])
def predict_batch(batch: BatchInput, db: Session = Depends(get_db)):
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.time()
    predictions = [run_single_prediction(app, c, db) for c in batch.customers]
    high_risk = sum(1 for p in predictions if p.risk_tier == "High")
    return BatchOutput(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
        processing_time_ms=round((time.time() - t0) * 1000, 1),
    )


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
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
        pred = run_single_prediction(app, customer, db)
        results.append({
            "customerID": pred.customer_id,
            "churn_probability": pred.churn_probability,
            "risk_tier": pred.risk_tier,
            "top_factor": pred.top_factors[0].feature if pred.top_factors else "N/A",
        })

    result_df = df.copy()
    result_df["churn_probability"] = [r["churn_probability"] for r in results]
    result_df["risk_tier"] = [r["risk_tier"] for r in results]
    result_df["top_factor"] = [r["top_factor"] for r in results]

    # Windows compatible temporary path
    out_path = os.path.join(tempfile.gettempdir(), "predictions_output.csv")
    result_df.to_csv(out_path, index=False)
    return FileResponse(out_path, media_type="text/csv", filename="predictions.csv")


@app.get("/drift/status", response_model=DriftStatusResponse)
def drift_status(db: Session = Depends(get_db)):
    latest = get_latest_drift(db)
    if not latest:
        return DriftStatusResponse(
            drift_detected=False, drift_score=0.0,
            status="healthy", last_checked=None, report_available=False
        )
    score = latest.drift_score or 0.0
    status = "healthy" if score < 0.1 else ("mild_drift" if score < 0.2 else "significant_drift")
    return DriftStatusResponse(
        drift_detected=bool(latest.drift_detected),
        drift_score=score, status=status,
        last_checked=latest.created_at, report_available=True,
    )


@app.get("/drift/report")
def drift_report(db: Session = Depends(get_db)):
    latest = get_latest_drift(db)
    if not latest:
        raise HTTPException(status_code=404, detail="No drift report available yet")
    return FileResponse(latest.report_path, media_type="text/html")


def _run_drift_check(db):
    """Internal: run Evidently drift check and log result."""
    import uuid
    records = last_n_inputs(db, n=500)
    if len(records) < 50:
        return
    # Reconstruct minimal dataframe from logged data — use hashes only
    # In production, store anonymized feature snapshots separately
    pass  # Full implementation would store feature snapshots at prediction time
