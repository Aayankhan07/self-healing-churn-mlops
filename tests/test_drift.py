"""Unit tests for data drift report generation."""

import pandas as pd
from src.monitor import generate_drift_report


def test_generate_drift_report_execution(sample_dataframe):
    reference_data = pd.concat([sample_dataframe] * 5, ignore_index=True)
    current_data = pd.concat([sample_dataframe] * 2, ignore_index=True)

    res = generate_drift_report(current_data, reference_data)
    assert isinstance(res, dict)
    assert "drift_detected" in res
    assert "drift_score" in res
    assert "report_path" in res
    assert res["n_samples"] == 2
