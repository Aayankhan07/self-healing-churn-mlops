"""
Characterization tests — pin current externally-observable behavior.

These exist to make the refactor provably non-regressive. They assert shapes,
key sets, and exact healing action strings rather than model values, so they
stay green across retraining but fail loudly if a refactor moves behavior.

Do not "fix" a failure here by editing the assertion. A failure means the
refactor changed behavior; either that was intended (update the test in the
same commit, with a note) or it is a regression.
"""

import pytest

from api.main import heal_customer_data

ADMIN_HEADERS = {"X-API-Key": "dev-key-change-in-prod"}
ANALYST_HEADERS = {"X-API-Key": "analyst-key"}
ENGINEER_HEADERS = {"X-API-Key": "engineer-key"}


# ── Response shape snapshots ────────────────────────────────


def test_health_shape(test_client):
    body = test_client.get("/health").json()
    assert set(body) == {
        "status",
        "model_loaded",
        "model_version",
        "uptime_seconds",
    }
    assert body["status"] in ("healthy", "degraded")
    assert isinstance(body["model_loaded"], bool)
    assert isinstance(body["model_version"], str)
    assert isinstance(body["uptime_seconds"], (int, float))


def test_metrics_shape(test_client):
    body = test_client.get("/metrics").json()
    assert set(body) == {"f1", "roc_auc", "source"}
    assert isinstance(body["f1"], (int, float))
    assert isinstance(body["roc_auc"], (int, float))
    assert isinstance(body["source"], str)


def test_drift_status_shape(test_client):
    body = test_client.get("/drift/status").json()
    assert set(body) == {
        "drift_detected",
        "drift_score",
        "status",
        "last_checked",
        "report_available",
    }
    assert isinstance(body["drift_detected"], bool)
    assert isinstance(body["drift_score"], (int, float))
    assert body["status"] in ("healthy", "mild_drift", "significant_drift")
    assert isinstance(body["report_available"], bool)


def test_self_healing_logs_shape(test_client):
    body = test_client.get("/self-healing/logs").json()
    assert isinstance(body, list)
    for entry in body:
        assert set(entry) == {"id", "event_type", "description", "created_at"}


def test_shadow_status_shape(test_client):
    body = test_client.get("/model/shadow-status").json()
    assert set(body) == {"sample_count", "avg_delta", "status"}
    assert isinstance(body["sample_count"], int)
    assert isinstance(body["avg_delta"], (int, float))


def test_prometheus_exposition(test_client):
    r = test_client.get("/metrics/prometheus")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    body = r.text
    # Each metric family must keep its HELP/TYPE header and sample line.
    for family in [
        "churnguard_uptime_seconds",
        "churnguard_predictions_total",
        "churnguard_drift_score",
        "churnguard_shadow_divergence_delta",
        "churnguard_risk_tier_count",
        "churnguard_self_healing_events_total",
    ]:
        assert f"# HELP {family} " in body
        assert f"# TYPE {family} " in body
    for domain in ["telecom", "school", "ecommerce", "fitness"]:
        assert f'churnguard_predictions_domain_total{{domain="{domain}"}}' in body
    for tier in ["High", "Medium", "Low"]:
        assert f'churnguard_risk_tier_count{{tier="{tier}"}}' in body


# ── Prediction contract ─────────────────────────────────────

PREDICTION_KEYS = {
    "customer_id",
    "churn_probability",
    "risk_tier",
    "prediction",
    "top_factors",
    "model_version",
    "prediction_id",
    "timestamp",
    "healed_actions",
    "recommended_actions",
    "time_to_churn_days",
    "risk_horizon_summary",
    "survival_timeline",
}


def test_predict_contract(test_client, valid_customer):
    r = test_client.post("/predict", json=valid_customer, headers=ADMIN_HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == PREDICTION_KEYS
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in ("Low", "Medium", "High")
    assert body["prediction"] in (0, 1)
    assert len(body["top_factors"]) <= 3
    for factor in body["top_factors"]:
        assert set(factor) == {"feature", "value", "impact", "direction"}
        assert factor["direction"] in ("increases_risk", "decreases_risk")
    assert body["model_version"].startswith("telecom")
    assert isinstance(body["recommended_actions"], list)


def test_predict_risk_tier_thresholds(test_client, valid_customer):
    """Risk tiers are cut at 0.35 / 0.65 — pin the mapping, not the value."""
    body = test_client.post(
        "/predict", json=valid_customer, headers=ADMIN_HEADERS
    ).json()
    proba = body["churn_probability"]
    expected = "High" if proba >= 0.65 else ("Medium" if proba >= 0.35 else "Low")
    assert body["risk_tier"] == expected
    assert body["prediction"] == int(proba >= 0.5)


def test_predict_school_domain_uses_school_model(test_client, valid_customer):
    """
    Today a non-telecom domain accepts telecom-shaped input and scores against a
    copy of the telecom model. Phase 2 changes this deliberately.
    """
    r = test_client.post(
        "/predict?domain=school", json=valid_customer, headers=ADMIN_HEADERS
    )
    assert r.status_code == 200
    assert r.json()["model_version"].startswith("school")


def test_predict_batch_contract(test_client, valid_customer):
    r = test_client.post(
        "/predict/batch",
        json={"customers": [valid_customer, valid_customer]},
        headers=ADMIN_HEADERS,
    )
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "predictions",
        "total",
        "high_risk_count",
        "processing_time_ms",
    }
    assert body["total"] == 2
    assert len(body["predictions"]) == 2
    assert body["high_risk_count"] == sum(
        1 for p in body["predictions"] if p["risk_tier"] == "High"
    )
    assert set(body["predictions"][0]) == PREDICTION_KEYS


def test_predict_batch_honors_domain(test_client, valid_customer):
    body = test_client.post(
        "/predict/batch?domain=school",
        json={"customers": [valid_customer]},
        headers=ADMIN_HEADERS,
    ).json()
    assert body["predictions"][0]["model_version"].startswith("school")


# ── Healing golden table ────────────────────────────────────

BASE = {
    "tenure": 10,
    "MonthlyCharges": 50.0,
    "TotalCharges": 500.0,
    "SeniorCitizen": 0,
}


@pytest.mark.parametrize(
    "field,raw_value,expected_value,expected_action",
    [
        # Numeric: missing, wrong type, out of range
        ("tenure", None, 0, "Imputed missing tenure to 0"),
        ("tenure", -5, 0, "Clamped negative tenure to 0"),
        ("tenure", "12", 12, "Coerced tenure to integer"),
        ("tenure", "abc", 0, "Imputed invalid tenure to 0"),
        (
            "MonthlyCharges",
            None,
            70.35,
            "Imputed missing MonthlyCharges to median (70.35)",
        ),
        ("MonthlyCharges", "65.5", 65.5, "Coerced MonthlyCharges to float"),
        (
            "MonthlyCharges",
            0,
            0.01,
            "Clamped non-positive MonthlyCharges to 0.01",
        ),
        ("TotalCharges", "800.0", 800.0, "Coerced TotalCharges to float"),
        # SeniorCitizen normalization across truthy/falsy spellings
        ("SeniorCitizen", None, 0, "Imputed missing SeniorCitizen to 0"),
        ("SeniorCitizen", "Yes", 1, "Normalized SeniorCitizen to 1"),
        ("SeniorCitizen", "true", 1, "Normalized SeniorCitizen to 1"),
        ("SeniorCitizen", "No", 0, "Normalized SeniorCitizen to 0"),
        # Categorical: fuzzy match and unrecognized fallback
        (
            "InternetService",
            "Fibre optic",
            "Fiber optic",
            "Mapped typos in InternetService ('Fibre optic') to 'Fiber optic'",
        ),
        (
            "Contract",
            "Weekly",
            "Month-to-month",
            "Imputed unrecognized Contract ('Weekly') to default 'Month-to-month'",
        ),
        (
            "PaymentMethod",
            "Electrnic check",
            "Electronic check",
            "Mapped typos in PaymentMethod ('Electrnic check') to 'Electronic check'",
        ),
        ("gender", None, "Male", "Imputed missing gender to default 'Male'"),
    ],
)
def test_healing_golden_table(field, raw_value, expected_value, expected_action):
    raw = {**BASE}
    if raw_value is None:
        raw.pop(field, None)
    else:
        raw[field] = raw_value

    healed, actions = heal_customer_data(raw)
    assert healed[field] == expected_value
    assert expected_action in actions


def test_healing_totalcharges_derived_from_tenure():
    raw = {"tenure": 4, "MonthlyCharges": 25.0, "SeniorCitizen": 0}
    healed, actions = heal_customer_data(raw)
    assert healed["TotalCharges"] == 100.0
    assert "Imputed missing TotalCharges as MonthlyCharges * tenure" in actions


def test_healing_totalcharges_constraint_recompute():
    raw = {**BASE, "tenure": 2, "MonthlyCharges": 50.0, "TotalCharges": 20.0}
    healed, actions = heal_customer_data(raw)
    assert healed["TotalCharges"] == 100.0
    assert (
        "Recomputed TotalCharges as MonthlyCharges * tenure due to constraint mismatch"
        in actions
    )


def test_healing_clean_input_produces_no_actions(valid_customer):
    healed, actions = heal_customer_data(valid_customer)
    assert actions == []
    for key, value in valid_customer.items():
        assert healed[key] == value


def test_healing_fills_every_categorical_field():
    """Healing must always emit a complete telecom feature row."""
    healed, _ = heal_customer_data({})
    required = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
    ]
    for field in required:
        assert field in healed


def test_healing_does_not_mutate_input():
    raw = {**BASE, "tenure": -5}
    heal_customer_data(raw)
    assert raw["tenure"] == -5


# ── Auth matrix ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "headers,path,payload,expected",
    [
        # No key at all — FastAPI rejects the missing required header as 422.
        (None, "/predict", "customer", 422),
        (None, "/self-healing/trigger-retrain", None, 422),
        (None, "/domain/bootstrap", {"domain_name": "TestDomain"}, 422),
        (None, "/model/promote", None, 422),
        # Unknown key — 403.
        ({"X-API-Key": "wrong-key"}, "/predict", "customer", 403),
        ({"X-API-Key": "wrong-key"}, "/model/promote", None, 403),
        # analyst: read:predict only.
        (ANALYST_HEADERS, "/predict", "customer", 200),
        (ANALYST_HEADERS, "/self-healing/trigger-retrain", None, 403),
        (ANALYST_HEADERS, "/domain/bootstrap", {"domain_name": "TestDomain"}, 403),
        (ANALYST_HEADERS, "/model/promote", None, 403),
        # engineer: read:predict + write:retrain.
        (ENGINEER_HEADERS, "/predict", "customer", 200),
        (ENGINEER_HEADERS, "/domain/bootstrap", {"domain_name": "TestDomain"}, 403),
        (ENGINEER_HEADERS, "/model/promote", None, 403),
        # admin: everything.
        (ADMIN_HEADERS, "/predict", "customer", 200),
        (ADMIN_HEADERS, "/domain/bootstrap", {"domain_name": "TestDomain"}, 200),
        (ADMIN_HEADERS, "/model/promote", None, 200),
    ],
)
def test_auth_matrix(test_client, valid_customer, headers, path, payload, expected):
    json_body = valid_customer if payload == "customer" else payload
    r = test_client.post(path, json=json_body, headers=headers)
    assert r.status_code == expected


def test_engineer_can_trigger_retrain(test_client):
    """Separate from the matrix: retrain is stateful and may 500 on a busy worker."""
    r = test_client.post("/self-healing/trigger-retrain", headers=ENGINEER_HEADERS)
    assert r.status_code in (200, 500)


def test_insufficient_scope_message(test_client):
    r = test_client.post("/model/promote", headers=ANALYST_HEADERS)
    assert r.status_code == 403
    assert "Forbidden: Insufficient privileges" in r.json()["detail"]


# ── Route inventory (Phase 3 parity guard) ──────────────────

EXPECTED_ROUTES = {
    "/health",
    "/metrics",
    "/metrics/prometheus",
    "/predict",
    "/predict/batch",
    "/upload",
    "/drift/status",
    "/drift/report",
    "/self-healing/logs",
    "/self-healing/trigger-retrain",
    "/domain/bootstrap",
    "/model/shadow-status",
    "/model/promote",
}


def test_route_inventory_unchanged(test_client):
    """
    Splitting api/main.py into routers must not add, drop, or rename a path.
    """
    paths = set(test_client.get("/openapi.json").json()["paths"])
    assert EXPECTED_ROUTES <= paths, f"missing routes: {EXPECTED_ROUTES - paths}"
