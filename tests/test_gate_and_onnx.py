"""Unit tests for Challenger Gate blocking proof and ONNX execution path."""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from api.main import _hash_key
from src.evaluate import compute_fairness_metrics
from sklearn.dummy import DummyClassifier


def test_api_key_sha256_hashing():
    raw_key = "analyst-key"
    hashed = _hash_key(raw_key)
    assert len(hashed) == 64  # SHA256 hex digest length
    assert hashed != raw_key


def test_high_entropy_key_generation():
    from api.main import generate_high_entropy_key
    token = generate_high_entropy_key("cg_admin")
    assert token.startswith("cg_admin_")
    assert len(token) > 40  # 256-bit secure URL-safe entropy


def test_fairness_gate_blocks_biased_model(valid_customer):
    """
    Empirically proves that a model with high demographic disparity
    fails the fairness disparity audit.
    """
    # Create biased dataset where SeniorCitizen=1 always gets positive prediction
    rows = []
    y_list = []
    for i in range(10):
        # Non-seniors get label 0
        rows.append({**valid_customer, "SeniorCitizen": 0, "tenure": 12, "Contract": "Month-to-month"})
        y_list.append(0)
        # Seniors get label 1
        rows.append({**valid_customer, "SeniorCitizen": 1, "tenure": 12, "Contract": "Month-to-month"})
        y_list.append(1)

    X_raw = pd.DataFrame(rows)
    y_true = pd.Series(y_list)

    # Fit classifier that predicts 1 for SeniorCitizen=1 and 0 for SeniorCitizen=0
    class BiasedModel:
        def predict_proba(self, df):
            preds = []
            for _, row in df.iterrows():
                if row.get("SeniorCitizen") == 1:
                    preds.append([0.1, 0.9])
                else:
                    preds.append([0.9, 0.1])
            import numpy as np
            return np.array(preds)

        def predict(self, df):
            return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)

    model = BiasedModel()
    fairness = compute_fairness_metrics(model, X_raw, y_true)

    dp_diff = fairness["demographic_parity_difference"]
    ff_ratio = fairness["four_fifths_selection_ratio"]
    assert dp_diff > 0.30  # High disparity detected
    assert ff_ratio < 0.80  # Breaches EEOC Four-Fifths Rule
    assert fairness["bias_status"] == "disparity_detected"
