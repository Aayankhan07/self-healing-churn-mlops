"""Unit and integration tests for self-healing features."""

import json
import threading
from api.main import heal_customer_data
from api.database import SessionLocal


def test_heal_customer_data_numeric():
    # 1. Test negative tenure
    raw_data = {
        "tenure": -5,
        "MonthlyCharges": 50.0,
        "TotalCharges": 100.0,
        "SeniorCitizen": 0,
    }
    healed, actions = heal_customer_data(raw_data)
    assert healed["tenure"] == 0
    assert "Clamped negative tenure to 0" in actions

    # 2. Test constraint TotalCharges < MonthlyCharges when tenure > 0
    raw_data = {
        "tenure": 2,
        "MonthlyCharges": 50.0,
        "TotalCharges": 20.0,
        "SeniorCitizen": 0,
    }
    healed, actions = heal_customer_data(raw_data)
    assert healed["TotalCharges"] == 100.0
    assert (
        "Recomputed TotalCharges as MonthlyCharges * tenure due to constraint mismatch"
        in actions
    )


def test_heal_customer_data_categorical():
    # Test spelling typo in PaymentMethod
    raw_data = {
        "tenure": 10,
        "MonthlyCharges": 50.0,
        "TotalCharges": 500.0,
        "SeniorCitizen": 0,
        "PaymentMethod": "Electrnic check",  # typo
    }
    healed, actions = heal_customer_data(raw_data)
    assert healed["PaymentMethod"] == "Electronic check"
    assert (
        "Mapped typos in PaymentMethod ('Electrnic check') to 'Electronic check'"
        in actions
    )


def test_heal_customer_data_senior_citizen():
    # Test senior citizen string representation
    raw_data = {
        "tenure": 10,
        "MonthlyCharges": 50.0,
        "TotalCharges": 500.0,
        "SeniorCitizen": "Yes",
    }
    healed, actions = heal_customer_data(raw_data)
    assert healed["SeniorCitizen"] == 1
    assert "Normalized SeniorCitizen to 1" in actions


def test_predict_endpoint_self_healing(test_client, valid_customer):
    # Submit a malformed input to the endpoint
    malformed = {
        **valid_customer,
        "tenure": -12,
        "TotalCharges": 10.0,  # too low relative to tenure and monthly charges
        "PaymentMethod": "Electrnic check",
    }
    r = test_client.post(
        "/predict", json=malformed, headers={"X-API-Key": "dev-key-change-in-prod"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["healed_actions"]
    assert "Clamped negative tenure to 0" in body["healed_actions"]
    assert (
        "Mapped typos in PaymentMethod ('Electrnic check') to 'Electronic check'"
        in body["healed_actions"]
    )


def test_self_healing_logs_endpoint(test_client):
    r = test_client.get("/self-healing/logs")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_database_self_healing_before_retraining(monkeypatch):
    from unittest.mock import MagicMock, patch
    from api.database import log_prediction, get_self_healing_logs
    from api.main import run_self_healing_retraining

    # 1. Create a clean database session
    db = SessionLocal()
    try:
        # Clear existing prediction logs to ensure deterministic tests
        from api.database import Prediction, SelfHealingLog

        db.query(Prediction).delete()
        db.query(SelfHealingLog).delete()
        db.commit()

        # 2. Insert an un-healed prediction record directly into the DB.
        unhealed_features = {
            "tenure": -10,
            "MonthlyCharges": 50.0,
            "TotalCharges": 20.0,
            "SeniorCitizen": "Yes",
            "PaymentMethod": "Electrnic check",
        }
        log_prediction(
            db=db,
            prediction_id="test-pred-id-999",
            customer_id="7590-VHVEG",
            input_hash="hash999",
            probability=0.9,
            risk_tier="High",
            prediction=1,
            model_ver="test-model-v1",
            features_json=json.dumps(unhealed_features),
            healed_actions=json.dumps([]),  # empty healed actions (as if logged raw)
        )
        db.commit()

        # 3. Mock dependencies of run_self_healing_retraining to avoid training a real model
        mock_app = MagicMock()
        mock_app.state.model_lock = threading.Lock()
        mock_app.state.model_container = {}

        with patch("src.train.train") as mock_train, patch("joblib.load") as mock_load:

            mock_load.return_value = MagicMock()

            # Run the retraining function which triggers database self-healing
            run_self_healing_retraining(mock_app)

            # Ensure it attempted to reload the model but skipped training or finished successfully
            assert mock_train.called

        # 4. Query the self-healing event logs and verify that database-level healing was logged
        sh_logs = get_self_healing_logs(db, limit=10)
        db_healing_events = [
            log for log in sh_logs if "Database-level self-healing" in log.description
        ]

        assert len(db_healing_events) > 0
        event_desc = db_healing_events[0].description
        assert "Clamped negative tenure to 0" in event_desc
        assert "Normalized SeniorCitizen to 1" in event_desc
        assert "Mapped typos in PaymentMethod" in event_desc
    finally:
        db.close()
