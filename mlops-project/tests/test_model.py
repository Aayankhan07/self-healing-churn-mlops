import os
import sys
import pytest
from fastapi.testclient import TestClient

# Add project root to sys.path to allow imports from api and src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.main import app
from src.data_prep import prepare_data
from src.train import train

@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Ensure that the model is built and ready for testing."""
    model_path = os.path.join("data", "model.pkl")
    if not os.path.exists(model_path):
        print("\nModel not found. Running data preparation and training dynamically for test environment...")
        prepare_data()
        train()

def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "message" in response.json()
        assert response.json()["status"] == "active"

def test_health_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

def test_prediction_endpoint():
    with TestClient(app) as client:
        payload = {
            "tenure": 3,
            "MonthlyCharges": 105.4,
            "TotalCharges": 316.2,
            "Contract": "Month-to-month",
            "InternetService": "Fiber optic",
            "TechSupport": "No",
            "OnlineSecurity": "No",
            "PaperlessBilling": "Yes"
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "churn_risk_level" in data
        assert "attrition_warning" in data
        
        # This is a high-risk customer profile, so we expect high risk level or churn prediction
        assert data["churn_risk_level"] in ["High", "Medium"]

def test_monitoring_endpoint():
    with TestClient(app) as client:
        # 1. Clean up or ensure a request log exists by making at least 5 prediction calls
        payloads = [
            {"tenure": 12, "MonthlyCharges": 65.5, "TotalCharges": 786.0, "Contract": "Month-to-month", "InternetService": "Fiber optic", "TechSupport": "No", "OnlineSecurity": "No", "PaperlessBilling": "Yes"},
            {"tenure": 34, "MonthlyCharges": 90.2, "TotalCharges": 3066.8, "Contract": "One year", "InternetService": "Fiber optic", "TechSupport": "Yes", "OnlineSecurity": "No", "PaperlessBilling": "Yes"},
            {"tenure": 2, "MonthlyCharges": 20.1, "TotalCharges": 40.2, "Contract": "Month-to-month", "InternetService": "No", "TechSupport": "No internet service", "OnlineSecurity": "No internet service", "PaperlessBilling": "No"},
            {"tenure": 72, "MonthlyCharges": 115.8, "TotalCharges": 8337.6, "Contract": "Two year", "InternetService": "Fiber optic", "TechSupport": "Yes", "OnlineSecurity": "Yes", "PaperlessBilling": "Yes"},
            {"tenure": 8, "MonthlyCharges": 45.3, "TotalCharges": 362.4, "Contract": "Month-to-month", "InternetService": "DSL", "TechSupport": "No", "OnlineSecurity": "Yes", "PaperlessBilling": "No"}
        ]
        
        for pl in payloads:
            res = client.post("/predict", json=pl)
            assert res.status_code == 200
            
        # 2. Trigger data drift monitoring check
        response = client.get("/monitor")
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        # If evidently is installed, it compiles the report and returns success
        if data["status"] == "success":
            assert "drift_detected" in data
            assert "drift_summary" in data
            assert "serving_records_analyzed" in data
            assert data["serving_records_analyzed"] >= 5
            
            # Check if the HTML report is downloadable
            report_res = client.get("/monitor/report")
            assert report_res.status_code == 200
            assert b"html" in report_res.content or b"<!DOCTYPE html>" in report_res.content
        else:
            # If evidently is missing from environment, it should state so gracefully
            assert "installed" in data["message"]
