"""
Drift detection service tests.

run_drift_check decides whether a domain's recent traffic has drifted and, if
so, whether to launch a retrain. It runs on a background thread with no
response to inspect, so these tests drive it directly and assert on what it
wrote to the database and whether it claimed the retraining slot.
"""

import json
import threading
import uuid
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from api.database import (
    DriftReport,
    Prediction,
    SelfHealingLog,
    SessionLocal,
    log_prediction,
)
from api.services import drift_service, model_registry

# run_drift_check ignores any domain with fewer than this many usable records.
MIN_RECORDS = 50


@pytest.fixture
def clean_db():
    db = SessionLocal()
    try:
        db.query(Prediction).delete()
        db.query(DriftReport).delete()
        db.query(SelfHealingLog).delete()
        db.commit()
        yield db
    finally:
        db.query(Prediction).delete()
        db.query(DriftReport).delete()
        db.query(SelfHealingLog).delete()
        db.commit()
        db.close()


@pytest.fixture
def fake_app():
    """Minimal stand-in for the FastAPI app: just the state the service uses."""
    return SimpleNamespace(
        state=SimpleNamespace(
            model_lock=threading.Lock(),
            retraining_status="idle",
            model_registry={},
        )
    )


def _seed_predictions(db, count, domain_id="telecom", features=None):
    features = features or {
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "SeniorCitizen": 0,
    }
    for i in range(count):
        log_prediction(
            db,
            str(uuid.uuid4()),
            f"CUST-{i}",
            "hash",
            0.5,
            "Medium",
            0,
            "telecom-v1",
            features_json=json.dumps(features),
            domain_id=domain_id,
        )


def _drift_result(score, n_samples=60):
    return {
        "drift_detected": score >= 0.2,
        "drift_score": score,
        "report_path": "reports/test_drift.html",
        "n_samples": n_samples,
    }


def test_too_few_records_is_a_no_op(clean_db, fake_app):
    """A domain without enough traffic has nothing to compare."""
    _seed_predictions(clean_db, MIN_RECORDS - 1)

    with patch.object(drift_service, "generate_drift_report") as report:
        drift_service.run_drift_check(fake_app, "telecom")

    assert not report.called
    assert clean_db.query(DriftReport).count() == 0


def test_records_without_features_do_not_count(clean_db, fake_app):
    """Rows with no features_json cannot be reconstructed into a frame."""
    for _ in range(MIN_RECORDS + 10):
        log_prediction(
            clean_db,
            str(uuid.uuid4()),
            "C",
            "hash",
            0.5,
            "Medium",
            0,
            "telecom-v1",
            features_json=None,
            domain_id="telecom",
        )

    with patch.object(drift_service, "generate_drift_report") as report:
        drift_service.run_drift_check(fake_app, "telecom")

    assert not report.called


def test_malformed_features_json_is_skipped(clean_db, fake_app):
    """One corrupt row must not abort the whole check."""
    _seed_predictions(clean_db, MIN_RECORDS + 5)
    log_prediction(
        clean_db,
        str(uuid.uuid4()),
        "BAD",
        "hash",
        0.5,
        "Medium",
        0,
        "telecom-v1",
        features_json="{not valid json",
        domain_id="telecom",
    )

    with patch.object(
        drift_service, "generate_drift_report", return_value=_drift_result(0.05)
    ) as report:
        drift_service.run_drift_check(fake_app, "telecom")

    assert report.called
    assert clean_db.query(DriftReport).count() == 1


def test_no_drift_logs_a_report_without_retraining(clean_db, fake_app):
    _seed_predictions(clean_db, MIN_RECORDS + 5)

    with patch.object(
        drift_service, "generate_drift_report", return_value=_drift_result(0.05)
    ):
        with patch.object(drift_service.threading, "Thread") as thread:
            drift_service.run_drift_check(fake_app, "telecom")

    report = clean_db.query(DriftReport).one()
    assert report.drift_detected == 0
    assert report.drift_score == 0.05
    assert report.domain_id == "telecom"
    assert not thread.called
    assert fake_app.state.retraining_status == "idle"


def test_drift_above_threshold_launches_retraining(clean_db, fake_app):
    _seed_predictions(clean_db, MIN_RECORDS + 5)

    with patch.object(
        drift_service, "generate_drift_report", return_value=_drift_result(0.42)
    ):
        with patch.object(drift_service.threading, "Thread") as thread:
            drift_service.run_drift_check(fake_app, "telecom")

    report = clean_db.query(DriftReport).one()
    assert report.drift_detected == 1
    assert thread.called
    # The slot is claimed so a second check cannot start a parallel retrain.
    assert fake_app.state.retraining_status == "running"

    events = [e.description for e in clean_db.query(SelfHealingLog).all()]
    assert any("Triggering automated retraining" in e for e in events)


def test_drift_while_already_retraining_does_not_start_a_second_run(clean_db, fake_app):
    _seed_predictions(clean_db, MIN_RECORDS + 5)
    fake_app.state.retraining_status = "running"

    with patch.object(
        drift_service, "generate_drift_report", return_value=_drift_result(0.42)
    ):
        with patch.object(drift_service.threading, "Thread") as thread:
            drift_service.run_drift_check(fake_app, "telecom")

    assert not thread.called
    events = [e.description for e in clean_db.query(SelfHealingLog).all()]
    assert any("already running" in e for e in events)


def test_report_failure_is_logged_not_raised(clean_db, fake_app):
    """A drift check runs on a background thread; it must never crash it."""
    _seed_predictions(clean_db, MIN_RECORDS + 5)

    with patch.object(
        drift_service,
        "generate_drift_report",
        side_effect=RuntimeError("evidently exploded"),
    ):
        drift_service.run_drift_check(fake_app, "telecom")

    events = [e.description for e in clean_db.query(SelfHealingLog).all()]
    assert any("Error running drift check" in e for e in events)
    assert any("evidently exploded" in e for e in events)


def test_check_is_scoped_to_its_domain(clean_db, fake_app):
    """
    Another domain's traffic must not satisfy this domain's record threshold —
    the bug that let a school drift check read telecom rows.
    """
    _seed_predictions(clean_db, MIN_RECORDS + 20, domain_id="telecom")
    _seed_predictions(clean_db, 5, domain_id="school")

    with patch.object(drift_service, "generate_drift_report") as report:
        drift_service.run_drift_check(fake_app, "school")

    assert not report.called


def test_domain_name_is_sanitized(clean_db, fake_app):
    """Callers may pass a display name; records are keyed by the sanitized id."""
    _seed_predictions(clean_db, MIN_RECORDS + 5, domain_id="telecom")

    with patch.object(
        drift_service, "generate_drift_report", return_value=_drift_result(0.05)
    ) as report:
        drift_service.run_drift_check(fake_app, "Telecom Customer Churn")

    assert report.called
    assert clean_db.query(DriftReport).one().domain_id == "telecom"


def test_claim_and_release_round_trip(fake_app):
    assert model_registry.claim_retraining_slot(fake_app) is True
    assert model_registry.claim_retraining_slot(fake_app) is False
    model_registry.release_retraining_slot(fake_app)
    assert model_registry.claim_retraining_slot(fake_app) is True
