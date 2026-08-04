"""Unit test suite for Fairness & Demographic Bias Analysis."""

import pandas as pd
import pytest
from src.evaluate import compute_fairness_metrics
from sklearn.dummy import DummyClassifier


def test_compute_fairness_metrics(valid_customer):
    rows = [
        {**valid_customer, "SeniorCitizen": 0, "Contract": "Month-to-month", "tenure": 5},
        {**valid_customer, "SeniorCitizen": 1, "Contract": "Month-to-month", "tenure": 10},
        {**valid_customer, "SeniorCitizen": 0, "Contract": "One year", "tenure": 30},
        {**valid_customer, "SeniorCitizen": 1, "Contract": "Two year", "tenure": 48},
        {**valid_customer, "SeniorCitizen": 0, "Contract": "Month-to-month", "tenure": 8},
        {**valid_customer, "SeniorCitizen": 1, "Contract": "One year", "tenure": 20},
    ]
    X_raw = pd.DataFrame(rows)
    y_true = pd.Series([1, 1, 0, 0, 1, 0])

    # Fit dummy classifier
    model = DummyClassifier(strategy="constant", constant=1)
    model.fit(X_raw, y_true)

    fairness = compute_fairness_metrics(model, X_raw, y_true)

    assert "subgroups" in fairness
    assert "SeniorCitizen" in fairness["subgroups"]
    assert "Contract" in fairness["subgroups"]
    assert "TenureBucket" in fairness["subgroups"]
    assert "demographic_parity_difference" in fairness
    assert "equalized_odds_difference" in fairness
    assert fairness["bias_status"] in ["acceptable", "disparity_detected"]
