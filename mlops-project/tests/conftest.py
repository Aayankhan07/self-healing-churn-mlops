"""Shared fixtures for all test files."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from api.main import app  # noqa: E402


@pytest.fixture(scope="session")
def test_client():
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
