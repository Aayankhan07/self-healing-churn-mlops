"""Unit tests for data drift report generation."""

from pathlib import Path

import pandas as pd
import pytest

import src.monitor as monitor
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


@pytest.fixture
def forced_evidently(monkeypatch):
    """
    Exercise the real-Evidently branch regardless of whether the package
    imports on this platform.

    Evidently is incompatible with some Python versions, so locally the module
    takes a mock-fallback path while CI takes the real one. A bug reachable
    only on the real path is therefore invisible during local development —
    which is exactly how the current_data=None crash reached CI.
    """
    if monitor.EVIDENTLY_AVAILABLE:
        yield
        return

    class FakeReport:
        def __init__(self, metrics):
            self.metrics = metrics

        def run(self, reference_data, current_data, column_mapping):
            assert current_data is not None, "current_data must be a DataFrame"
            assert reference_data is not None
            self.rows = len(current_data)

        def save_html(self, path):
            Path(path).write_text("<html>report</html>", encoding="utf-8")

        def as_dict(self):
            return {
                "metrics": [
                    {
                        "result": {
                            "dataset_drift": False,
                            "share_of_drifted_columns": 0.0,
                        }
                    }
                ]
            }

    monkeypatch.setattr(monitor, "EVIDENTLY_AVAILABLE", True)
    monkeypatch.setattr(monitor, "Report", FakeReport, raising=False)
    monkeypatch.setattr(monitor, "DataDriftPreset", lambda: None, raising=False)
    monkeypatch.setattr(
        monitor,
        "ColumnMapping",
        lambda target=None, prediction=None: None,
        raising=False,
    )
    yield


def test_report_regenerates_without_current_data(forced_evidently):
    """
    /drift/report passes current_data=None to regenerate a report on demand,
    before any traffic has been scored. That must not raise.
    """
    res = generate_drift_report(None, domain_id="telecom")
    assert res["drift_score"] == 0.0
    assert res["n_samples"] > 0
    assert res["report_path"]


def test_report_with_explicit_reference(forced_evidently, sample_dataframe):
    reference = pd.concat([sample_dataframe] * 4, ignore_index=True)
    current = pd.concat([sample_dataframe] * 3, ignore_index=True)

    res = generate_drift_report(current, reference, domain_id="telecom")
    assert res["n_samples"] == 3
    assert res["drift_detected"] is False
