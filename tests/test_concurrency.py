"""
Regression tests for the concurrency fixes in Phase 1.

Both cover races that only appear under simultaneous requests, so they drive the
underlying primitives directly rather than hoping a timing window reproduces.
"""

import threading
from io import StringIO

import api.main as main
from api.services import model_registry


def test_only_one_thread_claims_the_retraining_slot():
    """
    Bug #6: retraining_status was read and written outside the lock, so two
    concurrent drift checks could both observe "idle" and both launch training.
    """
    main.app.state.model_lock = threading.Lock()
    main.app.state.retraining_status = "idle"

    claims = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()  # maximize overlap on the check-and-set
        claims.append(model_registry.claim_retraining_slot(main.app))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(claims) == 1, "exactly one caller may claim the retraining slot"
    assert main.app.state.retraining_status == "running"

    main.app.state.retraining_status = "idle"


def test_claim_is_refused_while_running():
    main.app.state.model_lock = threading.Lock()
    main.app.state.retraining_status = "running"
    try:
        assert model_registry.claim_retraining_slot(main.app) is False
    finally:
        main.app.state.retraining_status = "idle"


def test_concurrent_uploads_get_distinct_files(test_client, tmp_path):
    """
    Bug #8: /upload wrote every caller's results to one fixed temp path, so
    overlapping uploads served each other's rows.
    """
    import pandas as pd

    from api.schemas import CustomerInput

    columns = [c for c in CustomerInput.model_fields if c != "customerID"]

    def make_csv(tenure):
        row = {
            "tenure": tenure,
            "MonthlyCharges": 65.0,
            "TotalCharges": 1560.0,
            "SeniorCitizen": 0,
            "gender": "Male",
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
        }
        path = tmp_path / f"batch_{tenure}.csv"
        pd.DataFrame([row])[columns].to_csv(path, index=False)
        return path

    results = {}

    def upload(tenure):
        path = make_csv(tenure)
        with open(path, "rb") as fh:
            r = test_client.post(
                "/upload",
                files={"file": (path.name, fh, "text/csv")},
                headers={"X-API-Key": "dev-key-change-in-prod"},
            )
        results[tenure] = r

    threads = [threading.Thread(target=upload, args=(t,)) for t in (5, 40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert set(results) == {5, 40}
    for tenure, r in results.items():
        assert r.status_code == 200
        returned = pd.read_csv(StringIO(r.text))
        # Each caller must get back its own row, not the other upload's.
        assert returned["tenure"].tolist() == [tenure]
        assert "churn_probability" in returned.columns
        assert "risk_tier" in returned.columns
