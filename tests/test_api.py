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


def test_admin_endpoints_require_auth(test_client):
    # Missing API Key
    r_retrain = test_client.post("/self-healing/trigger-retrain")
    assert r_retrain.status_code == 422

    r_bootstrap = test_client.post(
        "/domain/bootstrap", json={"domain_name": "TestDomain"}
    )
    assert r_bootstrap.status_code == 422

    r_promote = test_client.post("/model/promote")
    assert r_promote.status_code == 422

    # Invalid API Key
    bad_headers = {"X-API-Key": "invalid-key"}
    assert (
        test_client.post(
            "/self-healing/trigger-retrain", headers=bad_headers
        ).status_code
        == 403
    )
    assert (
        test_client.post(
            "/domain/bootstrap", json={"domain_name": "TestDomain"}, headers=bad_headers
        ).status_code
        == 403
    )
    assert test_client.post("/model/promote", headers=bad_headers).status_code == 403

    # Valid API Key
    good_headers = {"X-API-Key": "dev-key-change-in-prod"}
    assert test_client.post(
        "/self-healing/trigger-retrain", headers=good_headers
    ).status_code in [200, 500]
    assert (
        test_client.post(
            "/domain/bootstrap",
            json={"domain_name": "TestDomain"},
            headers=good_headers,
        ).status_code
        == 200
    )
    assert test_client.post("/model/promote", headers=good_headers).status_code == 200


def test_rbac_scopes(test_client, valid_customer):
    # 1. Analyst key: has 'read:predict' scope only
    analyst_headers = {"X-API-Key": "analyst-key"}
    r_pred = test_client.post("/predict", json=valid_customer, headers=analyst_headers)
    assert r_pred.status_code == 200

    r_admin = test_client.post("/model/promote", headers=analyst_headers)
    assert r_admin.status_code == 403
    assert "Forbidden: Insufficient privileges" in r_admin.json()["detail"]

    # 2. Engineer key: has 'read:predict' and 'write:retrain' scopes
    eng_headers = {"X-API-Key": "engineer-key"}
    assert (
        test_client.post(
            "/predict", json=valid_customer, headers=eng_headers
        ).status_code
        == 200
    )
    assert test_client.post(
        "/self-healing/trigger-retrain", headers=eng_headers
    ).status_code in [200, 500]

    r_eng_admin = test_client.post("/model/promote", headers=eng_headers)
    assert r_eng_admin.status_code == 403

    # 3. Admin key: has all scopes
    admin_headers = {"X-API-Key": "dev-key-change-in-prod"}
    assert (
        test_client.post(
            "/predict", json=valid_customer, headers=admin_headers
        ).status_code
        == 200
    )
    assert test_client.post("/model/promote", headers=admin_headers).status_code == 200
