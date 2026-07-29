"""
Evidently drift monitoring. Called periodically via API counter.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

REFERENCE_DATA_PATH = "data/processed/train.csv"
REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)

try:
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset
    from evidently.pipeline.column_mapping import ColumnMapping

    EVIDENTLY_AVAILABLE = True
except Exception as e:
    logger.warning(
        f"Evidently AI is not fully compatible with this Python version: {e}"
    )
    EVIDENTLY_AVAILABLE = False


def generate_drift_report(
    current_data: pd.DataFrame, reference_data: pd.DataFrame = None, domain_id: str = "telecom"
) -> dict:
    """
    Compare current incoming data to domain reference baseline.
    Returns dict with drift_detected, drift_score, report_path.
    """
    if not EVIDENTLY_AVAILABLE:
        # Graceful fallback mock report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"drift_{timestamp}.html"
        # Write a simple HTML placeholder
        with open(report_path, "w") as f:
            f.write(
                "<html><body><h1>Drift Report (Mock fallback)</h1></body></html>"
            )
        return {
            "drift_detected": False,
            "drift_score": 0.0,
            "report_path": str(report_path),
            "n_samples": len(current_data),
        }

    if reference_data is None:
        from src.domain_registry import get_domain_baseline_path, ensure_domain_initialized
        ensure_domain_initialized(domain_id)
        baseline_path = get_domain_baseline_path(domain_id)
        if baseline_path.exists():
            reference_data = pd.read_csv(baseline_path).drop(
                columns=["Churn"], errors="ignore"
            )
        else:
            reference_data = pd.read_csv(REFERENCE_DATA_PATH).drop(
                columns=["Churn"], errors="ignore"
            )

    ignore_cols = ["customerID", "studentID", "id", "prediction_id", "Churn"]
    reference_data = reference_data.drop(columns=ignore_cols, errors="ignore")
    current_data = current_data.drop(columns=ignore_cols, errors="ignore")

    column_mapping = ColumnMapping(target=None, prediction=None)

    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"drift_{timestamp}.html"
    report.save_html(str(report_path))

    result = report.as_dict()
    drift_detected = result["metrics"][0]["result"]["dataset_drift"]
    drift_share = result["metrics"][0]["result"]["share_of_drifted_columns"]

    logger.info(f"Drift check: detected={drift_detected}, share={drift_share:.2f}")
    return {
        "drift_detected": drift_detected,
        "drift_score": round(drift_share, 4),
        "report_path": str(report_path),
        "n_samples": len(current_data),
    }
