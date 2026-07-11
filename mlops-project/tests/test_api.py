"""API endpoint tests using FastAPI TestClient."""


def test_health_check(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    assert "status" in r.json()


def test_predict_valid_input(test_client, valid_customer):
    r = test_client.post(
        "/predict", json=valid_customer, headers={"X-API-Key": "dev-key-change-in-prod"}
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in ["Low", "Medium", "High"]
    assert len(body["top_factors"]) <= 3


def test_predict_missing_api_key(test_client, valid_customer):
    r = test_client.post("/predict", json=valid_customer)
    assert r.status_code == 422  # missing header


def test_predict_invalid_api_key(test_client, valid_customer):
    r = test_client.post(
        "/predict", json=valid_customer, headers={"X-API-Key": "wrong-key"}
    )
    assert r.status_code == 403


def test_predict_invalid_tenure(test_client, valid_customer):
    invalid = {**valid_customer, "tenure": -1}
    r = test_client.post(
        "/predict", json=invalid, headers={"X-API-Key": "dev-key-change-in-prod"}
    )
    assert r.status_code == 200
    assert "Clamped negative tenure to 0" in r.json()["healed_actions"]


def test_predict_invalid_contract(test_client, valid_customer):
    invalid = {**valid_customer, "Contract": "Weekly"}
    r = test_client.post(
        "/predict", json=invalid, headers={"X-API-Key": "dev-key-change-in-prod"}
    )
    assert r.status_code == 200
    assert (
        "Imputed unrecognized Contract ('Weekly') to default 'Month-to-month'"
        in r.json()["healed_actions"]
    )


def test_drift_status_endpoint(test_client):
    r = test_client.get("/drift/status")
    assert r.status_code == 200
    assert "drift_detected" in r.json()
