"""
Monitoring route tests.

/metrics and /drift/* have several fallback layers each — domain file, MLflow,
root file, hardcoded default — and the tests that existed only ever hit the
first branch. These drive each layer, since a fallback nobody exercises is a
fallback nobody knows is broken.
"""

import json
import uuid

import pytest

from api.database import DriftReport, SessionLocal, log_drift_report
from src.domain_registry import get_domain_model_dir

ADMIN_HEADERS = {"X-API-Key": "dev-key-change-in-prod"}


@pytest.fixture
def clean_drift_reports():
    db = SessionLocal()
    try:
        db.query(DriftReport).delete()
        db.commit()
        yield db
    finally:
        db.query(DriftReport).delete()
        db.commit()
        db.close()


# ── /metrics ────────────────────────────────────────────────


def test_metrics_prefers_the_domain_file(test_client, tmp_path, monkeypatch):
    domain_dir = get_domain_model_dir("telecom")
    metrics_file = domain_dir / "eval_metrics.json"
    original = metrics_file.read_text() if metrics_file.exists() else None
    metrics_file.write_text(json.dumps({"f1": 0.9111, "roc_auc": 0.9555}))

    try:
        body = test_client.get("/metrics?domain=telecom").json()
        assert body["f1"] == 0.911
        assert body["roc_auc"] == 0.956
        assert "telecom" in body["source"]
    finally:
        if original is not None:
            metrics_file.write_text(original)
        else:
            metrics_file.unlink(missing_ok=True)


def test_metrics_accepts_alternate_key_names(test_client):
    """Older metric files wrote test_f1 / test_auc instead of f1 / roc_auc."""
    domain_dir = get_domain_model_dir("telecom")
    metrics_file = domain_dir / "eval_metrics.json"
    original = metrics_file.read_text() if metrics_file.exists() else None
    metrics_file.write_text(json.dumps({"test_f1": 0.77, "test_auc": 0.88}))

    try:
        body = test_client.get("/metrics?domain=telecom").json()
        assert body["f1"] == 0.77
        assert body["roc_auc"] == 0.88
    finally:
        if original is not None:
            metrics_file.write_text(original)
        else:
            metrics_file.unlink(missing_ok=True)


def test_metrics_survives_a_corrupt_domain_file(test_client):
    """A malformed file falls through to the next source rather than 500ing."""
    domain_dir = get_domain_model_dir("telecom")
    metrics_file = domain_dir / "eval_metrics.json"
    original = metrics_file.read_text() if metrics_file.exists() else None
    metrics_file.write_text("{ this is not json")

    try:
        r = test_client.get("/metrics?domain=telecom")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"f1", "roc_auc", "source"}
    finally:
        if original is not None:
            metrics_file.write_text(original)
        else:
            metrics_file.unlink(missing_ok=True)


def test_metrics_falls_back_for_an_unknown_domain(test_client):
    """A domain with no metrics anywhere still answers with the default."""
    body = test_client.get("/metrics?domain=custom_never_trained").json()
    assert isinstance(body["f1"], (int, float))
    assert isinstance(body["roc_auc"], (int, float))
    assert body["source"]


# ── /drift/status ───────────────────────────────────────────


def test_drift_status_with_no_history(test_client, clean_drift_reports):
    body = test_client.get("/drift/status?domain=telecom").json()
    assert body["drift_detected"] is False
    assert body["drift_score"] == 0.0
    assert body["status"] == "healthy"
    assert body["last_checked"] is None
    assert body["report_available"] is False


@pytest.mark.parametrize(
    "score,expected_status",
    [
        (0.0, "healthy"),
        (0.09, "healthy"),
        (0.1, "mild_drift"),
        (0.19, "mild_drift"),
        (0.2, "significant_drift"),
        (0.85, "significant_drift"),
    ],
)
def test_drift_status_thresholds(
    test_client, clean_drift_reports, score, expected_status
):
    """Status bands are cut at 0.1 and 0.2."""
    log_drift_report(
        clean_drift_reports,
        report_id=str(uuid.uuid4()),
        report_path="reports/does_not_exist.html",
        drift_detected=int(score >= 0.2),
        drift_score=score,
        n_samples=100,
        domain_id="telecom",
    )

    body = test_client.get("/drift/status?domain=telecom").json()
    assert body["status"] == expected_status
    assert body["drift_score"] == score
    assert body["report_available"] is True


def test_drift_status_is_scoped_by_domain(test_client, clean_drift_reports):
    """A telecom report must not show up as school's drift status."""
    log_drift_report(
        clean_drift_reports,
        report_id=str(uuid.uuid4()),
        report_path="reports/x.html",
        drift_detected=1,
        drift_score=0.9,
        n_samples=100,
        domain_id="telecom",
    )

    assert test_client.get("/drift/status?domain=school").json()["status"] == "healthy"
    assert (
        test_client.get("/drift/status?domain=telecom").json()["status"]
        == "significant_drift"
    )


# ── /drift/report ───────────────────────────────────────────


def test_drift_report_generates_one_when_none_exists(
    test_client, clean_drift_reports
):
    r = test_client.get("/drift/report?domain=telecom")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_drift_report_regenerates_when_the_file_is_gone(
    test_client, clean_drift_reports
):
    """A logged report whose file was deleted must not 404."""
    log_drift_report(
        clean_drift_reports,
        report_id=str(uuid.uuid4()),
        report_path="reports/deleted_by_cleanup.html",
        drift_detected=0,
        drift_score=0.01,
        n_samples=100,
        domain_id="telecom",
    )

    r = test_client.get("/drift/report?domain=telecom")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]


def test_drift_report_serves_a_real_report(
    test_client, clean_drift_reports, tmp_path
):
    report = tmp_path / "real_report.html"
    report.write_text(
        "<html><head><title>Evidently</title></head>"
        "<body><div id='root'>real content</div></body></html>",
        encoding="utf-8",
    )
    log_drift_report(
        clean_drift_reports,
        report_id=str(uuid.uuid4()),
        report_path=str(report),
        drift_detected=0,
        drift_score=0.02,
        n_samples=100,
        domain_id="telecom",
    )

    r = test_client.get("/drift/report?domain=telecom")
    assert r.status_code == 200
    assert "real content" in r.text


# ── /health ─────────────────────────────────────────────────


def test_health_for_an_unloaded_domain(test_client):
    """An unknown domain still answers rather than erroring."""
    body = test_client.get("/health?domain=custom_not_registered").json()
    assert body["status"] in ("healthy", "degraded")
    assert isinstance(body["model_version"], str)


def test_health_accepts_a_display_name(test_client):
    body = test_client.get("/health?domain=Telecom Customer Churn").json()
    assert body["model_version"].startswith("telecom")
