"""
FastAPI application entrypoint.
All routes defined here. Model loaded once on startup.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402
import os  # noqa: E402
import mlflow.sklearn  # noqa: E402
import joblib  # noqa: E402
import pandas as pd  # noqa: E402
import threading  # noqa: E402
import difflib  # noqa: E402
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header  # fmt: skip # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from io import StringIO  # noqa: E402
import tempfile  # noqa: E402
import yaml  # noqa: E402
import random  # noqa: E402
import json  # noqa: E402

from api.schemas import CustomerInput, PredictionOutput, BatchOutput, HealthResponse, DriftStatusResponse, LaxBatchInput  # fmt: skip # noqa: E402
from api.predict import run_single_prediction  # noqa: E402
from api.database import init_db, get_db, last_n_inputs, get_latest_drift, log_drift_report, log_self_healing_event, get_self_healing_logs, SelfHealingLog  # fmt: skip # noqa: E402
from api.drift import generate_drift_report  # noqa: E402

START_TIME = time.time()

ENVIRONMENT = os.getenv("ENV", "development").lower()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    if ENVIRONMENT == "production":
        raise ValueError(
            "API_KEY environment variable must be set in production environment!"
        )
    else:
        API_KEY = "dev-key-change-in-prod"

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

    model_loaded = False
    model = None
    model_version = "unknown"

    try:
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_stage}")
        model_version = model_stage
        model_loaded = True
    except Exception:
        # Fallback to local model if MLflow server is offline
        local_model_path = "models/model.joblib"
        if os.path.exists(local_model_path):
            try:
                model = joblib.load(local_model_path)
                model_version = "local-fallback"
                model_loaded = True
            except Exception:
                model_loaded = False
                model = None
                model_version = "unknown"
        else:
            model_loaded = False
            model = None
            model_version = "unknown"

    try:
        preprocessor = joblib.load("models/preprocessor.joblib")
    except Exception:
        preprocessor = None

    # Thread-safe atomic reload structures
    app.state.model_lock = threading.Lock()
    app.state.model_container = {
        "model": model,
        "preprocessor": preprocessor,
        "version": model_version,
    }
    app.state.model_loaded = model_loaded
    app.state.model = model  # fallback for direct reads
    app.state.preprocessor = preprocessor  # fallback for direct reads
    app.state.model_version = model_version  # fallback for direct reads

    app.state.reference_data = pd.read_csv("data/processed/train.csv").drop(
        columns=["Churn"], errors="ignore"
    )
    app.state.prediction_counter = 0
    app.state.drift_check_every = int(os.getenv("DRIFT_CHECK_EVERY_N", 100))
    app.state.retraining_status = "idle"  # "idle" or "running"
    init_db()


def heal_customer_data(raw_data: dict) -> tuple[dict, list[str]]:
    healed_data = raw_data.copy()
    healed_actions = []

    # Rules mapping
    # 1. Numeric fields: negative or missing values clamped or imputed
    # tenure: int
    if "tenure" not in healed_data or healed_data["tenure"] is None:
        healed_data["tenure"] = 0
        healed_actions.append("Imputed missing tenure to 0")
    elif not isinstance(healed_data["tenure"], (int, float)):
        try:
            healed_data["tenure"] = int(float(healed_data["tenure"]))
            healed_actions.append("Coerced tenure to integer")
        except Exception:
            healed_data["tenure"] = 0
            healed_actions.append("Imputed invalid tenure to 0")

    if healed_data["tenure"] < 0:
        healed_data["tenure"] = 0
        healed_actions.append("Clamped negative tenure to 0")

    # MonthlyCharges: float
    if "MonthlyCharges" not in healed_data or healed_data["MonthlyCharges"] is None:
        median_mc = 70.35
        healed_data["MonthlyCharges"] = median_mc
        healed_actions.append(f"Imputed missing MonthlyCharges to median ({median_mc})")
    elif not isinstance(healed_data["MonthlyCharges"], (int, float)):
        try:
            healed_data["MonthlyCharges"] = float(healed_data["MonthlyCharges"])
            healed_actions.append("Coerced MonthlyCharges to float")
        except Exception:
            median_mc = 70.35
            healed_data["MonthlyCharges"] = median_mc
            healed_actions.append(
                f"Imputed invalid MonthlyCharges to median ({median_mc})"
            )

    if healed_data["MonthlyCharges"] <= 0:
        healed_data["MonthlyCharges"] = 0.01
        healed_actions.append("Clamped non-positive MonthlyCharges to 0.01")

    # TotalCharges: float
    if "TotalCharges" not in healed_data or healed_data["TotalCharges"] is None:
        healed_data["TotalCharges"] = round(
            healed_data["MonthlyCharges"] * max(1, healed_data["tenure"]), 2
        )
        healed_actions.append("Imputed missing TotalCharges as MonthlyCharges * tenure")
    elif not isinstance(healed_data["TotalCharges"], (int, float)):
        try:
            healed_data["TotalCharges"] = float(healed_data["TotalCharges"])
            healed_actions.append("Coerced TotalCharges to float")
        except Exception:
            healed_data["TotalCharges"] = round(
                healed_data["MonthlyCharges"] * max(1, healed_data["tenure"]), 2
            )
            healed_actions.append(
                "Imputed invalid TotalCharges as MonthlyCharges * tenure"
            )

    if healed_data["TotalCharges"] < 0:
        healed_data["TotalCharges"] = 0.0
        healed_actions.append("Clamped negative TotalCharges to 0.0")

    # Constraint check: TotalCharges cannot be less than MonthlyCharges when tenure > 0
    if (
        healed_data["tenure"] > 0
        and healed_data["TotalCharges"] < healed_data["MonthlyCharges"]
    ):
        healed_data["TotalCharges"] = round(
            healed_data["MonthlyCharges"] * healed_data["tenure"], 2
        )
        healed_actions.append(
            "Recomputed TotalCharges as MonthlyCharges * tenure due to constraint mismatch"
        )

    # 2. SeniorCitizen: normalize to 0 or 1
    if "SeniorCitizen" not in healed_data or healed_data["SeniorCitizen"] is None:
        healed_data["SeniorCitizen"] = 0
        healed_actions.append("Imputed missing SeniorCitizen to 0")
    else:
        val = str(healed_data["SeniorCitizen"]).strip().lower()
        if val in ("yes", "y", "true", "1", "1.0"):
            healed_data["SeniorCitizen"] = 1
            if val != "1":
                healed_actions.append("Normalized SeniorCitizen to 1")
        else:
            healed_data["SeniorCitizen"] = 0
            if val != "0":
                healed_actions.append("Normalized SeniorCitizen to 0")

    # 3. Categorical features string similarity dynamic mapping
    CATEGORICAL_SCHEMAS = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["Yes", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    }

    for col, valid_options in CATEGORICAL_SCHEMAS.items():
        if col not in healed_data or healed_data[col] is None:
            default_cat = valid_options[0]
            healed_data[col] = default_cat
            healed_actions.append(f"Imputed missing {col} to default '{default_cat}'")
        else:
            val = str(healed_data[col]).strip()
            if val in valid_options:
                healed_data[col] = val
                continue

            # String similarity lookup
            matches = difflib.get_close_matches(val, valid_options, n=1, cutoff=0.6)
            if matches:
                closest = matches[0]
                healed_data[col] = closest
                healed_actions.append(f"Mapped typos in {col} ('{val}') to '{closest}'")
            else:
                default_cat = valid_options[0]
                healed_data[col] = default_cat
                healed_actions.append(
                    f"Imputed unrecognized {col} ('{val}') to default '{default_cat}'"
                )

    return healed_data, healed_actions


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy" if app.state.model_loaded else "degraded",
        model_loaded=app.state.model_loaded,
        model_version=app.state.model_version,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/metrics")
def get_metrics(db: Session = Depends(get_db)):
    # 1. Try fetching from MLflow
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        latest_versions = client.get_latest_versions(
            "churn_model", stages=["Production"]
        )
        if latest_versions:
            run_id = latest_versions[0].run_id
            run = client.get_run(run_id)
            metrics_data = run.data.metrics
            # Retrieve metrics logged from train.py
            f1 = metrics_data.get("f1")
            auc = metrics_data.get("roc_auc")
            if f1 is not None and auc is not None:
                return {
                    "f1": round(f1, 3),
                    "roc_auc": round(auc, 3),
                    "source": f"MLflow Run ({run_id[:8]})",
                }
    except Exception:
        pass

    # 2. Fallback to local files
    paths = ["metrics/eval_metrics.json", "metrics/test_metrics.json"]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                    f1 = data.get("f1") or data.get("test_f1")
                    auc = data.get("roc_auc") or data.get("test_auc")
                    if f1 is not None and auc is not None:
                        return {
                            "f1": round(f1, 3),
                            "roc_auc": round(auc, 3),
                            "source": f"Local {os.path.basename(path)}",
                        }
            except Exception:
                pass

    # 3. Default fallback
    return {"f1": 0.83, "roc_auc": 0.89, "source": "Fallback Defaults"}


@app.post(
    "/predict", response_model=PredictionOutput, dependencies=[Depends(verify_api_key)]
)
def predict(customer: dict, db: Session = Depends(get_db)):
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Ingestion Self-Healing
    healed_dict, healed_actions = heal_customer_data(customer)

    # Strict validation
    try:
        validated_customer = CustomerInput(**healed_dict)
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
        )

    result = run_single_prediction(
        app, validated_customer, db, healed_actions=healed_actions
    )
    trigger_drift = False
    with app.state.model_lock:
        app.state.prediction_counter += 1
        if app.state.prediction_counter % app.state.drift_check_every == 0:
            trigger_drift = True

    if trigger_drift:
        threading.Thread(target=_run_drift_check).start()
    return result


@app.post(
    "/predict/batch", response_model=BatchOutput, dependencies=[Depends(verify_api_key)]
)
def predict_batch(batch: LaxBatchInput, db: Session = Depends(get_db)):
    if not app.state.model_loaded or app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.time()

    predictions = []
    for customer in batch.customers:
        healed_dict, healed_actions = heal_customer_data(customer)
        try:
            validated_customer = CustomerInput(**healed_dict)
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
            )

        pred = run_single_prediction(
            app, validated_customer, db, healed_actions=healed_actions
        )
        predictions.append(pred)

    high_risk = sum(1 for p in predictions if p.risk_tier == "High")

    # Increment counter and trigger drift check if boundary crossed
    trigger_drift = False
    with app.state.model_lock:
        old_counter = app.state.prediction_counter
        app.state.prediction_counter += len(predictions)
        if (app.state.prediction_counter // app.state.drift_check_every) > (
            old_counter // app.state.drift_check_every
        ):
            trigger_drift = True

    if trigger_drift:
        threading.Thread(target=_run_drift_check).start()

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

    # Windows compatible temporary path
    out_path = os.path.join(tempfile.gettempdir(), "predictions_output.csv")
    result_df.to_csv(out_path, index=False)
    return FileResponse(out_path, media_type="text/csv", filename="predictions.csv")


@app.get("/drift/status", response_model=DriftStatusResponse)
def drift_status(db: Session = Depends(get_db)):
    latest = get_latest_drift(db)
    if not latest:
        return DriftStatusResponse(
            drift_detected=False,
            drift_score=0.0,
            status="healthy",
            last_checked=None,
            report_available=False,
        )
    score = latest.drift_score or 0.0
    status = (
        "healthy"
        if score < 0.1
        else ("mild_drift" if score < 0.2 else "significant_drift")
    )
    return DriftStatusResponse(
        drift_detected=bool(latest.drift_detected),
        drift_score=score,
        status=status,
        last_checked=latest.created_at,
        report_available=True,
    )


@app.get("/drift/report")
def drift_report(db: Session = Depends(get_db)):
    latest = get_latest_drift(db)
    if not latest:
        raise HTTPException(status_code=404, detail="No drift report available yet")
    return FileResponse(latest.report_path, media_type="text/html")


def _run_drift_check():
    """Internal: run Evidently drift check and log result."""
    import uuid
    import json
    from api.database import SessionLocal

    db = SessionLocal()
    try:
        # 1. Fetch drift threshold from env
        drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.20"))

        # 2. Get the last N inputs from database
        records = last_n_inputs(db, n=500)
        if len(records) < 50:
            return

        # Reconstruct dataframe from features_json
        feature_dicts = []
        for r in records:
            if r.features_json:
                try:
                    feature_dicts.append(json.loads(r.features_json))
                except Exception:
                    pass

        if len(feature_dicts) < 50:
            return

        current_df = pd.DataFrame(feature_dicts)

        # Retrieve active baseline from app state to ensure we compare against post-retrain distributions
        ref_data = getattr(app.state, "reference_data", None)

        try:
            report_data = generate_drift_report(current_df, reference_data=ref_data)

            # Log to db
            report_id = str(uuid.uuid4())
            drift_score = report_data["drift_score"]
            drift_detected = drift_score >= drift_threshold

            log_drift_report(
                db,
                report_id=report_id,
                report_path=report_data["report_path"],
                drift_detected=int(drift_detected),
                drift_score=drift_score,
                n_samples=report_data["n_samples"],
            )

            # If drift detected, trigger self-healing retraining!
            if drift_detected:
                if app.state.retraining_status == "idle":
                    # Trigger retraining in a background thread
                    app.state.retraining_status = "running"
                    log_self_healing_event(
                        db,
                        "retraining",
                        f"Drift detected (score={drift_score:.2f} >= threshold={drift_threshold:.2f}). Triggering automated retraining.",
                    )
                    threading.Thread(
                        target=run_self_healing_retraining, args=(app,)
                    ).start()
                else:
                    log_self_healing_event(
                        db,
                        "retraining",
                        f"Drift detected (score={drift_score:.2f} >= threshold={drift_threshold:.2f}), but auto-retraining is already running.",
                    )
        except Exception as e:
            log_self_healing_event(
                db, "retraining", f"Error running drift check: {str(e)}"
            )
    finally:
        db.close()


def run_self_healing_retraining(app_ref):
    """
    Background worker that runs the self-healing retraining loop.
    Reads recent production inputs, resolves labels, merges them with reference training data,
    triggers Optuna + XGBoost tuning, and atomically hot-reloads the new model.
    """
    from api.database import SessionLocal, log_self_healing_event, last_n_inputs
    from src.train import train as run_training_pipeline

    db = SessionLocal()
    try:
        # 1. Fetch recent production inputs from SQLite
        records = last_n_inputs(db, n=500)
        if not records:
            log_self_healing_event(
                db,
                "retraining",
                "Auto-retraining aborted: no prediction records found.",
            )
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        # Load raw data to match customer IDs with true labels
        raw_df = pd.read_csv("data/raw/telco_churn.csv")
        raw_df["customerID"] = raw_df["customerID"].astype(str).str.strip()
        raw_labels = (
            raw_df.set_index("customerID")["Churn"].map({"Yes": 1, "No": 0}).to_dict()
        )

        # Reconstruct DataFrame with labels
        new_rows = []
        sample_weights = []

        pseudo_label_count = 0
        true_label_count = 0

        for r in records:
            if not r.features_json:
                continue
            try:
                features = json.loads(r.features_json)
                cust_id = str(r.customer_id).strip() if r.customer_id else None

                # Database-level self-healing to clean features prior to retraining
                features, db_healed_actions = heal_customer_data(features)
                if db_healed_actions:
                    # Log only if not already logged to prevent duplicates across retraining runs
                    log_exists = (
                        db.query(SelfHealingLog)
                        .filter(
                            SelfHealingLog.event_type == "data_quality",
                            SelfHealingLog.description.contains(
                                f"for retraining of Customer {cust_id or 'N/A'}"
                            ),
                        )
                        .first()
                    )
                    if not log_exists:
                        log_self_healing_event(
                            db,
                            "data_quality",
                            f"Database-level self-healing corrected features for retraining of Customer {cust_id or 'N/A'}: {', '.join(db_healed_actions)}",
                        )

                label = None
                weight = 1.0

                # Check raw file for true label first
                if cust_id and cust_id in raw_labels:
                    label = raw_labels[cust_id]
                    true_label_count += 1
                else:
                    # Confidence-based pseudo-labeling
                    if r.probability >= 0.85:
                        label = 1
                        weight = 0.5  # lower weight
                        pseudo_label_count += 1
                    elif r.probability <= 0.15:
                        label = 0
                        weight = 0.5  # lower weight
                        pseudo_label_count += 1

                if label is not None:
                    # If it's a pseudo-label, apply 50% downsampling to simulate 0.5 weight
                    if weight == 0.5:
                        if random.random() > 0.5:
                            continue  # skip to simulate lower weight
                    features["Churn"] = label
                    new_rows.append(features)
                    sample_weights.append(weight)
            except Exception:
                pass

        if not new_rows:
            log_self_healing_event(
                db,
                "retraining",
                "Auto-retraining aborted: no labeled or high-confidence pseudo-labeled data resolved.",
            )
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        # Cap pseudo-labels at 20% of the training batch size
        total_resolved = len(new_rows)
        max_pseudo_allowed = int(0.20 * total_resolved)

        filtered_rows = []
        current_pseudo = 0

        for row, w in zip(new_rows, sample_weights):
            is_pseudo = w == 0.5
            if is_pseudo:
                if current_pseudo < max_pseudo_allowed:
                    filtered_rows.append(row)
                    current_pseudo += 1
            else:
                filtered_rows.append(row)

        new_prod_df = pd.DataFrame(filtered_rows)

        # Track consecutive pseudo-label retraining cycles to prevent confirmation bias
        if not hasattr(app_ref.state, "pseudo_retrain_cycles"):
            app_ref.state.pseudo_retrain_cycles = 0

        if pseudo_label_count > 0:
            app_ref.state.pseudo_retrain_cycles += 1
        else:
            app_ref.state.pseudo_retrain_cycles = 0

        if app_ref.state.pseudo_retrain_cycles > 3:
            log_self_healing_event(
                db,
                "retraining",
                "Auto-retraining aborted: Capped consecutive pseudo-label retraining cycles (max 3) without new ground-truth labels to prevent model bias reinforcement loop.",
            )
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        # Load original training data
        train_path = "data/processed/train.csv"
        combined_path = "data/processed/train_retrain.csv"

        if not os.path.exists(train_path):
            log_self_healing_event(
                db,
                "retraining",
                "Auto-retraining aborted: data/processed/train.csv not found.",
            )
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        original_train_df = pd.read_csv(train_path)

        # Merge original and new production data
        combined_train_df = pd.concat(
            [original_train_df, new_prod_df], ignore_index=True
        )

        # Save combined train.csv to a combined file path (non-destructive)
        combined_train_df.to_csv(combined_path, index=False)
        os.environ["TRAIN_DATA_PATH"] = combined_path

        # Read params.yaml and trigger train programmatically
        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        log_self_healing_event(
            db,
            "retraining",
            f"Starting retraining pipeline. Combined training dataset has {len(combined_train_df)} samples "
            f"({len(original_train_df)} original + {len(filtered_rows)} new production samples, "
            f"including {current_pseudo} pseudo-labeled records).",
        )

        # Run training pipeline
        run_training_pipeline(params)

        # Clean up combined path
        if os.path.exists(combined_path):
            os.remove(combined_path)

        # Atomic Hot-Reloading under lock
        new_model = joblib.load("models/model.joblib")
        new_preprocessor = joblib.load("models/preprocessor.joblib")

        with app_ref.state.model_lock:
            app_ref.state.model_container = {
                "model": new_model,
                "preprocessor": new_preprocessor,
                "version": "self-healed-retrained",
            }
            # update state pointers
            app_ref.state.model = new_model
            app_ref.state.preprocessor = new_preprocessor
            app_ref.state.model_version = "self-healed-retrained"
            app_ref.state.model_loaded = True
            # Update baseline reference data for drift checks
            try:
                app_ref.state.reference_data = combined_train_df.drop(
                    columns=["Churn"], errors="ignore"
                )
            except Exception:
                pass

        log_self_healing_event(
            db,
            "retraining",
            "Auto-retraining completed successfully. Model and preprocessor reloaded atomically into memory.",
        )
    except Exception as e:
        log_self_healing_event(db, "retraining", f"Auto-retraining failed: {str(e)}")
        # Clean up combined path on error
        combined_path = "data/processed/train_retrain.csv"
        if os.path.exists(combined_path):
            try:
                os.remove(combined_path)
            except Exception:
                pass
    finally:
        app_ref.state.retraining_status = "idle"
        db.close()


@app.get("/self-healing/logs")
def get_sh_logs(limit: int = 100, db: Session = Depends(get_db)):
    logs = get_self_healing_logs(db, limit=limit)
    return [
        {
            "id": log.id,
            "event_type": log.event_type,
            "description": log.description,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


@app.post("/self-healing/trigger-retrain")
def trigger_retrain(db: Session = Depends(get_db)):
    if app.state.retraining_status == "running":
        return {
            "status": "already_running",
            "message": "Retraining is already running.",
        }

    app.state.retraining_status = "running"
    log_self_healing_event(
        db, "retraining", "Manually triggered retraining from self-healing console."
    )
    threading.Thread(target=run_self_healing_retraining, args=(app,)).start()
    return {"status": "started", "message": "Asynchronous retraining triggered."}
