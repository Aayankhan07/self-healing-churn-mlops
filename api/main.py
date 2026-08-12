"""
FastAPI application entrypoint.
All routes defined here. Model loaded once on startup.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import time  # noqa: E402
import os  # noqa: E402
import logging  # noqa: E402
import mlflow.sklearn  # noqa: E402
import pandas as pd  # noqa: E402
import threading  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Header  # fmt: skip # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from io import StringIO  # noqa: E402
import tempfile  # noqa: E402
import yaml  # noqa: E402
import random  # noqa: E402
import json  # noqa: E402
import hashlib  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from api.schemas import CustomerInput, PredictionOutput, BatchOutput, HealthResponse, DriftStatusResponse, LaxBatchInput, build_request_model  # fmt: skip # noqa: E402
from api.predict import run_single_prediction  # noqa: E402
from api.database import init_db, get_db, last_n_inputs, get_latest_drift, log_drift_report, log_self_healing_event, get_self_healing_logs  # fmt: skip # noqa: E402
from src.monitor import generate_drift_report  # noqa: E402
from src.domains import get_domain_spec  # noqa: E402
from src.healing import heal  # noqa: E402

START_TIME = time.time()

import secrets  # noqa: E402

ENVIRONMENT = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

ANALYST_SCOPES = ["read:predict"]
ENGINEER_SCOPES = ["read:predict", "write:retrain"]
ADMIN_SCOPES = ["read:predict", "write:retrain", "admin:bootstrap", "admin:promote"]

# Every role key must be supplied explicitly in production. Falling back to a
# shared or well-known default would silently grant admin scope to whoever
# holds the read key.
_DEV_DEFAULTS = {
    "API_KEY": "dev-key-change-in-prod",
    "API_KEY_ANALYST": "analyst-key",
    "API_KEY_ENGINEER": "engineer-key",
}


def _require_key(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    if IS_PRODUCTION:
        raise ValueError(
            f"{name} environment variable must be set in production environment!"
        )
    return _DEV_DEFAULTS[name]


API_KEY = _require_key("API_KEY")
API_KEY_ANALYST = _require_key("API_KEY_ANALYST")
API_KEY_ENGINEER = _require_key("API_KEY_ENGINEER")

API_KEY_ADMIN = os.getenv("API_KEY_ADMIN")
if not API_KEY_ADMIN:
    if IS_PRODUCTION:
        raise ValueError(
            "API_KEY_ADMIN environment variable must be set in production environment!"
        )
    API_KEY_ADMIN = API_KEY


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_high_entropy_key(prefix: str = "cg") -> str:
    """Utility generating 256-bit cryptographically secure high-entropy API token."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


# Hashed Role-Based Access Control (RBAC) Scope Mappings
HASHED_API_KEY_SCOPES = {
    _hash_key(API_KEY_ANALYST): ANALYST_SCOPES,
    _hash_key(API_KEY_ENGINEER): ENGINEER_SCOPES,
    _hash_key(API_KEY_ADMIN): ADMIN_SCOPES,
}


def verify_scope(required_scope: str):
    """Dependency factory verifying SHA-256 hashed API key privileges against required scope."""

    def scope_checker(x_api_key: str = Header(...)):
        hashed_input = _hash_key(x_api_key)
        scopes = HASHED_API_KEY_SCOPES.get(hashed_input)
        if not scopes:
            # Legacy single-key fallback. Outside production only: granting the
            # full admin scope to the generic API_KEY is exactly the escalation
            # path we refuse to ship.
            if not IS_PRODUCTION and secrets.compare_digest(
                hashed_input, _hash_key(API_KEY)
            ):
                scopes = ADMIN_SCOPES
            else:
                raise HTTPException(status_code=403, detail="Invalid API key")

        if required_scope not in scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Forbidden: Insufficient privileges. Required scope '{required_scope}'.",
            )
        return x_api_key

    return scope_checker


def verify_api_key(x_api_key: str = Header(...)):
    """Backwards-compatible dependency for read:predict scope."""
    return verify_scope("read:predict")(x_api_key)


from api.metrics_prometheus import router as prometheus_router  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    # Initialize thread lock and domain-isolated model registry
    app.state.model_lock = threading.Lock()
    app.state.model_registry = {}

    from src.domain_registry import ensure_domain_initialized
    import src.domain_registry as domain_registry

    default_domains = ["telecom", "school", "ecommerce", "fitness"]

    for d_id in default_domains:
        try:
            ensure_domain_initialized(d_id)
            d_model = domain_registry.load_domain_model(d_id)
            d_prep = domain_registry.load_domain_preprocessor(d_id)
            app.state.model_registry[d_id] = {
                "model": d_model,
                "preprocessor": d_prep,
                "version": f"{d_id}-v1",
            }
        except Exception as e:
            logger.warning(f"Could not load domain model '{d_id}': {e}")

    # Set default telecom model container for backward compatibility
    telecom_container = app.state.model_registry.get(
        "telecom", {"model": None, "preprocessor": None, "version": "unknown"}
    )
    app.state.model_container = telecom_container
    app.state.model = telecom_container["model"]
    app.state.preprocessor = telecom_container["preprocessor"]
    app.state.model_version = telecom_container["version"]
    app.state.model_loaded = telecom_container["model"] is not None

    app.state.reference_data = pd.read_csv("data/processed/train.csv").drop(
        columns=["Churn"], errors="ignore"
    )
    app.state.prediction_counter = 0
    app.state.drift_check_every = int(os.getenv("DRIFT_CHECK_EVERY_N", 100))
    app.state.retraining_status = "idle"
    init_db()

    yield


app = FastAPI(
    title="ChurnGuard API",
    description="Production-grade, self-healing MLOps churn prediction microservice",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(prometheus_router)


def heal_customer_data(
    raw_data: dict, domain_id: str = "telecom"
) -> tuple[dict, list[str]]:
    """
    Repair a dirty inbound record against its domain's schema.

    The rules themselves live in src/domains/; this stays as the entry point the
    routes call. Defaults to telecom so existing callers keep their behavior.
    """
    return heal(raw_data, get_domain_spec(domain_id))


@app.get("/health", response_model=HealthResponse)
def health(domain: str = "telecom"):
    from src.domain_registry import sanitize_domain_id

    domain_key = sanitize_domain_id(domain)
    domain_model = getattr(app.state, "model_registry", {}).get(domain_key)
    is_loaded = domain_model is not None or getattr(app.state, "model_loaded", False)
    version = (
        domain_model.get("version", f"{domain_key}-v1")
        if domain_model
        else getattr(app.state, "model_version", "v1.0.0")
    )
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_version=version,
        uptime_seconds=round(time.time() - START_TIME, 1),
    )


@app.get("/metrics")
def get_metrics(domain: str = "telecom", db: Session = Depends(get_db)):
    from src.domain_registry import sanitize_domain_id, get_domain_model_dir

    domain_key = sanitize_domain_id(domain)
    domain_metrics_path = get_domain_model_dir(domain_key) / "eval_metrics.json"

    if domain_metrics_path.exists():
        try:
            with open(domain_metrics_path) as f:
                data = json.load(f)
                f1 = data.get("f1") or data.get("test_f1")
                auc = data.get("roc_auc") or data.get("auc_roc") or data.get("test_auc")
                if f1 is not None and auc is not None:
                    return {
                        "f1": round(f1, 3),
                        "roc_auc": round(auc, 3),
                        "source": f"Domain {domain_key} Metrics",
                    }
        except Exception:
            pass

    # 1. Try fetching from MLflow
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        latest_versions = client.get_latest_versions(
            f"churn_model_{domain_key}", stages=["Production"]
        )
        if latest_versions:
            run_id = latest_versions[0].run_id
            run = client.get_run(run_id)
            metrics_data = run.data.metrics
            f1 = metrics_data.get("f1")
            auc = metrics_data.get("roc_auc") or metrics_data.get("auc_roc")
            if f1 is not None and auc is not None:
                return {
                    "f1": round(f1, 3),
                    "roc_auc": round(auc, 3),
                    "source": f"MLflow Run ({run_id[:8]})",
                }
    except Exception:
        pass

    # 2. Fallback to local root files
    paths = ["metrics/eval_metrics.json", "metrics/test_metrics.json"]
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    data = json.load(f)
                    f1 = data.get("f1") or data.get("test_f1")
                    auc = (
                        data.get("roc_auc")
                        or data.get("auc_roc")
                        or data.get("test_auc")
                    )
                    if f1 is not None and auc is not None:
                        return {
                            "f1": round(f1, 3),
                            "roc_auc": round(auc, 3),
                            "source": f"Local {os.path.basename(path)}",
                        }
            except Exception:
                pass

    # 3. Default fallback
    return {"f1": 0.83, "roc_auc": 0.89, "source": f"Default Fallback ({domain_key})"}


@app.post(
    "/predict", response_model=PredictionOutput, dependencies=[Depends(verify_api_key)]
)
def predict(customer: dict, domain: str = "telecom", db: Session = Depends(get_db)):
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

    result = run_single_prediction(
        app, validated_customer, db, healed_actions=healed_actions, domain_id=domain_val
    )
    trigger_drift = False
    with app.state.model_lock:
        app.state.prediction_counter += 1
        if app.state.prediction_counter % app.state.drift_check_every == 0:
            trigger_drift = True

    if trigger_drift:
        threading.Thread(target=_run_drift_check, args=(domain_val,)).start()
    return result


@app.post(
    "/predict/batch", response_model=BatchOutput, dependencies=[Depends(verify_api_key)]
)
def predict_batch(
    batch: LaxBatchInput, domain: str = "telecom", db: Session = Depends(get_db)
):
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
        threading.Thread(target=_run_drift_check, args=(domain,)).start()

    return BatchOutput(
        predictions=predictions,
        total=len(predictions),
        high_risk_count=high_risk,
        processing_time_ms=round((time.time() - t0) * 1000, 1),
    )


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload_csv(
    file: UploadFile = File(...), domain: str = "telecom", db: Session = Depends(get_db)
):
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


@app.get("/drift/status", response_model=DriftStatusResponse)
def drift_status(domain: str = "telecom", db: Session = Depends(get_db)):
    from src.domain_registry import sanitize_domain_id

    domain_key = sanitize_domain_id(domain)
    latest = get_latest_drift(db, domain_id=domain_key)
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
def drift_report(domain: str = "telecom", db: Session = Depends(get_db)):
    from src.domain_registry import sanitize_domain_id

    domain_key = sanitize_domain_id(domain)
    latest = get_latest_drift(db, domain_id=domain_key)
    if not latest or not os.path.exists(latest.report_path):
        from src.monitor import generate_drift_report

        res = generate_drift_report(None, domain_id=domain_key)
        return FileResponse(res["report_path"], media_type="text/html")

    with open(latest.report_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    if "Mock fallback" in content or "<html><body><h1>" in content:
        from src.monitor import generate_drift_report

        res = generate_drift_report(None, domain_id=domain_key)
        return FileResponse(res["report_path"], media_type="text/html")

    return FileResponse(latest.report_path, media_type="text/html")


def _claim_retraining_slot() -> bool:
    """
    Atomically claim the single retraining slot.

    Returns True if this caller transitioned the status from idle to running and
    is therefore responsible for launching (and eventually clearing) the retrain.
    Checking and setting under the lock is what stops two concurrent drift checks
    from both launching a training run.
    """
    with app.state.model_lock:
        if app.state.retraining_status != "idle":
            return False
        app.state.retraining_status = "running"
        return True


def _run_drift_check(domain_id: str = "telecom"):
    """Internal: run Evidently drift check against domain baseline and log result."""
    import uuid
    import json
    from api.database import SessionLocal
    from src.domain_registry import sanitize_domain_id

    domain_id = sanitize_domain_id(domain_id)
    db = SessionLocal()
    try:
        # 1. Fetch drift threshold from env
        drift_threshold = float(os.getenv("DRIFT_THRESHOLD", "0.20"))

        # 2. Get the last N inputs from database, scoped to this domain
        records = last_n_inputs(db, n=500, domain_id=domain_id)
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

        try:
            report_data = generate_drift_report(current_df, domain_id=domain_id)

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
                domain_id=domain_id,
            )

            # If drift detected, trigger self-healing retraining!
            if drift_detected:
                if _claim_retraining_slot():
                    log_self_healing_event(
                        db,
                        "retraining",
                        f"Drift detected (score={drift_score:.2f} >= threshold={drift_threshold:.2f}). Triggering automated retraining.",
                        domain_id=domain_id,
                    )
                    threading.Thread(
                        target=run_self_healing_retraining, args=(app, domain_id)
                    ).start()
                else:
                    log_self_healing_event(
                        db,
                        "retraining",
                        f"Drift detected (score={drift_score:.2f} >= threshold={drift_threshold:.2f}), but auto-retraining is already running.",
                        domain_id=domain_id,
                    )
        except Exception as e:
            log_self_healing_event(
                db,
                "retraining",
                f"Error running drift check: {str(e)}",
                domain_id=domain_id,
            )
    finally:
        db.close()


def run_self_healing_retraining(app_ref, domain_id: str = "telecom"):
    """
    Background worker that runs the self-healing retraining loop for a specific domain.
    Reads recent production inputs, resolves labels, merges them with domain baseline data,
    triggers Optuna + XGBoost tuning, and atomically hot-reloads the new domain model into memory.
    """
    from api.database import SessionLocal, log_self_healing_event, last_n_inputs
    from src.train import train as run_training_pipeline
    from src.domain_registry import (
        load_domain_model,
        load_domain_preprocessor,
        sanitize_domain_id,
    )

    domain_id = sanitize_domain_id(domain_id)
    os.environ["TARGET_DOMAIN"] = domain_id

    db = SessionLocal()
    try:
        # 1. Fetch recent production inputs from SQLite, scoped to this domain
        records = last_n_inputs(db, n=500, domain_id=domain_id)
        if not records:
            log_self_healing_event(
                db,
                "retraining",
                f"Auto-retraining aborted for domain '{domain_id}': no prediction records found.",
                domain_id=domain_id,
            )
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        # Resolve ground-truth labels from this domain's own label source. A
        # domain without one must not be retrained: every row would be a
        # pseudo-label derived from the model's own output, which trains the
        # model on its own opinions.
        spec = get_domain_spec(domain_id)
        raw_path = spec.label_source_path
        if not raw_path:
            log_self_healing_event(
                db,
                "retraining",
                f"Auto-retraining aborted for domain '{domain_id}': no ground-truth "
                f"label source is configured, so every label would be a pseudo-label.",
                domain_id=domain_id,
            )
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        raw_labels = {}
        if os.path.exists(raw_path):
            raw_df = pd.read_csv(raw_path)
            id_col = spec.id_column
            target = spec.target_column
            if id_col in raw_df.columns and target in raw_df.columns:
                raw_df[id_col] = raw_df[id_col].astype(str).str.strip()
                raw_labels = (
                    raw_df.set_index(id_col)[target]
                    .map({"Yes": 1, "No": 0})
                    .to_dict()
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

                features, db_healed_actions = heal(features, spec)
                if db_healed_actions:
                    log_self_healing_event(
                        db,
                        "data_quality",
                        f"Database-level self-healing corrected features for retraining of Customer {cust_id or 'N/A'}: {', '.join(db_healed_actions)}",
                        domain_id=domain_id,
                    )

                label = None
                weight = 1.0

                if cust_id and cust_id in raw_labels:
                    label = raw_labels[cust_id]
                    true_label_count += 1
                else:
                    # Enforce pseudo-label ratio cap (max 30% pseudo-labels) and weight decay (0.25)
                    MAX_PSEUDO_LABEL_RATIO = 0.30
                    current_pseudo_ratio = pseudo_label_count / max(
                        1, (true_label_count + pseudo_label_count)
                    )
                    if current_pseudo_ratio < MAX_PSEUDO_LABEL_RATIO:
                        if r.probability >= 0.85:
                            label = 1
                            weight = 0.25
                            pseudo_label_count += 1
                        elif r.probability <= 0.15:
                            label = 0
                            weight = 0.25
                            pseudo_label_count += 1

                if label is not None:
                    if weight == 0.25:
                        if random.random() > 0.5:
                            continue
                    features["Churn"] = label
                    new_rows.append(features)
                    sample_weights.append(weight)
            except Exception:
                pass

        if not new_rows:
            log_self_healing_event(
                db,
                "retraining",
                f"Auto-retraining aborted for domain '{domain_id}': no labeled records resolved.",
                domain_id=domain_id,
            )
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        new_prod_df = pd.DataFrame(new_rows)
        train_path = "data/processed/train.csv"
        combined_path = "data/processed/train_retrain.csv"

        if not os.path.exists(train_path):
            app_ref.state.retraining_status = "idle"
            db.close()
            return

        original_train_df = pd.read_csv(train_path)
        combined_train_df = pd.concat(
            [original_train_df, new_prod_df], ignore_index=True
        )
        combined_train_df.to_csv(combined_path, index=False)
        os.environ["TRAIN_DATA_PATH"] = combined_path

        with open("params.yaml") as f:
            params = yaml.safe_load(f)

        log_self_healing_event(
            db,
            "retraining",
            f"Starting retraining pipeline for domain '{domain_id}'. Combined dataset has {len(combined_train_df)} samples ({true_label_count} true labels, {pseudo_label_count} pseudo-labels).",
            domain_id=domain_id,
        )

        run_training_pipeline(params)

        if os.path.exists(combined_path):
            os.remove(combined_path)

        # Register retrained model as Challenger (Champion remains active until promoted)
        new_model = load_domain_model(domain_id)
        new_preprocessor = load_domain_preprocessor(domain_id)

        with app_ref.state.model_lock:
            domain_container = app_ref.state.model_registry.setdefault(domain_id, {})
            domain_container["challenger"] = {
                "model": new_model,
                "preprocessor": new_preprocessor,
                "version": f"{domain_id}-challenger-v2",
            }

        log_self_healing_event(
            db,
            "retraining",
            f"Auto-retraining completed for domain '{domain_id}'. Retrained model registered as Challenger in shadow evaluation mode.",
            domain_id=domain_id,
        )
        from src.notifications import send_slack_alert

        send_slack_alert(
            "retraining",
            f"Challenger Model Registered for domain {domain_id}",
            "Auto-retraining completed. Challenger version registered for shadow deployment evaluation.",
            domain_id=domain_id,
        )
    except Exception as e:
        log_self_healing_event(
            db,
            "retraining",
            f"Auto-retraining failed: {str(e)}",
            domain_id=domain_id,
        )
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
def get_sh_logs(
    domain: str = "telecom", limit: int = 100, db: Session = Depends(get_db)
):
    from src.domain_registry import sanitize_domain_id

    domain_key = sanitize_domain_id(domain)
    logs = get_self_healing_logs(db, limit=limit, domain_id=domain_key)
    return [
        {
            "id": log.id,
            "event_type": log.event_type,
            "description": log.description,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


@app.post(
    "/self-healing/trigger-retrain",
    dependencies=[Depends(verify_scope("write:retrain"))],
)
def trigger_retrain(domain: str = "telecom", db: Session = Depends(get_db)):
    from src.domain_registry import sanitize_domain_id

    domain_key = sanitize_domain_id(domain)
    if not _claim_retraining_slot():
        return {
            "status": "already_running",
            "message": f"Retraining is already running for domain '{domain_key}'.",
        }

    log_self_healing_event(
        db,
        "retraining",
        f"Manually triggered retraining for domain '{domain_key}' from self-healing console.",
        domain_id=domain_key,
    )
    threading.Thread(target=run_self_healing_retraining, args=(app, domain_key)).start()
    return {
        "status": "started",
        "message": f"Asynchronous retraining triggered for domain '{domain_key}'.",
    }


@app.post("/domain/bootstrap", dependencies=[Depends(verify_scope("admin:bootstrap"))])
def bootstrap_domain_endpoint(payload: dict):
    domain_name = payload.get("domain_name")
    if not domain_name:
        raise HTTPException(status_code=400, detail="domain_name is required")

    from src.domain_registry import (
        bootstrap_custom_domain,
        load_domain_model,
        load_domain_preprocessor,
    )

    domain_key = bootstrap_custom_domain(domain_name)

    # Bootstrapping writes the domain's baseline, which is what its spec is
    # inferred from — drop any spec cached before that file existed.
    from src.domains import reset_spec_cache

    reset_spec_cache()

    d_model = load_domain_model(domain_key)
    d_prep = load_domain_preprocessor(domain_key)

    with app.state.model_lock:
        app.state.model_registry[domain_key] = {
            "model": d_model,
            "preprocessor": d_prep,
            "version": f"{domain_key}-v1",
        }

    return {
        "status": "success",
        "domain_key": domain_key,
        "message": f"Domain '{domain_name}' bootstrapped and isolated.",
    }


@app.get("/model/shadow-status")
def shadow_status(domain: str = "telecom", db: Session = Depends(get_db)):
    from src.domain_registry import sanitize_domain_id

    domain_key = sanitize_domain_id(domain)
    from api.database import get_shadow_stats

    return get_shadow_stats(db, domain_id=domain_key)


@app.post("/model/promote", dependencies=[Depends(verify_scope("admin:promote"))])
def promote_challenger(domain: str = "telecom", db: Session = Depends(get_db)):
    from src.domain_registry import sanitize_domain_id

    domain_key = sanitize_domain_id(domain)
    domain_container = getattr(app.state, "model_registry", {}).get(domain_key)
    if not domain_container or "challenger" not in domain_container:
        return {
            "status": "info",
            "message": f"No challenger model active for domain '{domain_key}'. Current champion is active.",
        }

    with app.state.model_lock:
        domain_container["champion"] = domain_container["challenger"]
        domain_container["model"] = domain_container["challenger"]["model"]
        domain_container["preprocessor"] = domain_container["challenger"][
            "preprocessor"
        ]
        domain_container["version"] = domain_container["challenger"]["version"]

        # Synchronize ONNX runtime engine upon model promotion
        try:
            from src.onnx_exporter import ONNXInferenceEngine
            from src.domain_registry import get_domain_model_dir

            onnx_path = str(get_domain_model_dir(domain_key) / "model.onnx")
            if "onnx_engine" in domain_container["challenger"]:
                domain_container["onnx_engine"] = domain_container["challenger"][
                    "onnx_engine"
                ]
            elif Path(onnx_path).exists():
                domain_container["onnx_engine"] = ONNXInferenceEngine(onnx_path)
        except Exception:
            pass

        del domain_container["challenger"]

    from api.database import log_self_healing_event

    log_self_healing_event(
        db,
        "retraining",
        f"Promoted Challenger model to Champion for domain '{domain_key}'.",
        domain_id=domain_key,
    )

    from src.notifications import send_slack_alert

    send_slack_alert(
        "promotion",
        f"Model Promoted for domain {domain_key}",
        f"Challenger model version {domain_container['version']} promoted to Champion.",
        domain_id=domain_key,
    )

    return {
        "status": "success",
        "message": f"Challenger promoted to Champion for domain '{domain_key}'.",
    }
