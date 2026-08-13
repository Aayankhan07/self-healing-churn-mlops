"""
Drift detection.

Compares recent production inputs for a domain against that domain's baseline
and, when the drift score crosses the configured threshold, hands off to the
retraining service.
"""

import logging
import os
import threading

import pandas as pd

from api.database import last_n_inputs, log_drift_report, log_self_healing_event
from api.services import model_registry
from api.services.retrain_service import run_self_healing_retraining
from src.monitor import generate_drift_report

logger = logging.getLogger(__name__)


def run_drift_check(app, domain_id: str = "telecom"):
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
                if model_registry.claim_retraining_slot(app):
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
