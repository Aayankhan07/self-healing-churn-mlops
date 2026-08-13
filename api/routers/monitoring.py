"""
Monitoring routes: service health, model quality metrics, and drift reports.
"""

import json
import logging
import os
import time

from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from api.database import get_db, get_latest_drift
from api.schemas import DriftStatusResponse, HealthResponse
from src.domain_registry import get_domain_model_dir, sanitize_domain_id
from src.domains import get_domain_spec

logger = logging.getLogger(__name__)
router = APIRouter()

# Process start, not import time of this module — uptime must survive the split.
START_TIME = time.time()


@router.get("/health", response_model=HealthResponse)
def health(request: Request, domain: str = "telecom"):
    app = request.app
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
        demo_fixture=get_domain_spec(domain_key).is_demo_fixture,
    )


@router.get("/metrics")
def get_metrics(domain: str = "telecom", db: Session = Depends(get_db)):
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


@router.get("/drift/status", response_model=DriftStatusResponse)
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


@router.get("/drift/report")
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
