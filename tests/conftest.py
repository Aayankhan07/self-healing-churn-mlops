"""Shared fixtures for all test files."""

from pathlib import Path

import numpy as np
import pytest
import pandas as pd
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Artifacts the API needs at startup. They are produced by the DVC pipeline and
# are deliberately gitignored, so a clean checkout — CI in particular — has
# none of them. Rather than committing a trained model and 600KB of data, the
# suite synthesizes equivalents on demand.
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

TELECOM_CATEGORIES = {
    "gender": ["Male", "Female"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}


def _synthetic_telecom_frame(n_rows: int = 300) -> pd.DataFrame:
    """A telecom-shaped dataset with a learnable signal."""
    rng = np.random.default_rng(20260813)

    frame = pd.DataFrame(
        {
            "tenure": rng.integers(0, 72, n_rows),
            "MonthlyCharges": rng.uniform(20.0, 120.0, n_rows).round(2),
            "SeniorCitizen": rng.integers(0, 2, n_rows),
        }
    )
    frame["TotalCharges"] = (
        frame["MonthlyCharges"] * frame["tenure"].clip(lower=1)
    ).round(2)

    for column, options in TELECOM_CATEGORIES.items():
        frame[column] = rng.choice(options, n_rows)

    # Churn correlates with short tenure, high charges, and month-to-month —
    # the model needs a signal to fit, not noise.
    risk = (
        (frame["tenure"] < 12).astype(int)
        + (frame["MonthlyCharges"] > 80).astype(int)
        + (frame["Contract"] == "Month-to-month").astype(int)
    )
    frame["Churn"] = (risk + rng.normal(0, 0.5, n_rows) >= 2).astype(int)
    # Guarantee both classes are present whatever the draw.
    frame.loc[frame.index[:20], "Churn"] = 1
    frame.loc[frame.index[20:40], "Churn"] = 0
    return frame


def _ensure_processed_splits() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if (PROCESSED_DIR / "train.csv").exists():
        return

    frame = _synthetic_telecom_frame()
    train, val, test = frame[:200], frame[200:250], frame[250:]
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)


def _ensure_telecom_model() -> None:
    """Fit a small pipeline so the API has something to serve."""
    telecom_dir = MODELS_DIR / "telecom"
    if (telecom_dir / "model.joblib").exists():
        return

    import joblib
    from imblearn.pipeline import Pipeline as ImbPipeline
    from xgboost import XGBClassifier

    from src.features import build_preprocessor, engineer_features

    frame = pd.read_csv(PROCESSED_DIR / "train.csv")
    y = frame["Churn"]
    X = engineer_features(frame.drop(columns=["Churn"]))

    preprocessor = build_preprocessor()
    # No SMOTE and few trees: this exists to be loadable and fast, not accurate.
    pipeline = ImbPipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=10,
                    max_depth=3,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(X, y)

    telecom_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, telecom_dir / "model.joblib")
    joblib.dump(preprocessor, telecom_dir / "preprocessor.joblib")

    # ensure_domain_initialized() seeds other domains from these root copies.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODELS_DIR / "model.joblib")
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")


@pytest.fixture(scope="session", autouse=True)
def provision_artifacts():
    """
    Make sure the data splits and a loadable model exist before anything
    imports the app. Real artifacts are left untouched when present, so a
    developer's trained model is never overwritten.
    """
    _ensure_processed_splits()
    _ensure_telecom_model()


@pytest.fixture(scope="session")
def test_client(provision_artifacts):
    from api.main import app

    with TestClient(app) as client:
        yield client


@pytest.fixture
def valid_customer():
    return {
        "tenure": 24,
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


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        [
            {
                "tenure": 24,
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
        ]
    )
