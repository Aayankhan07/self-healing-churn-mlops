"""Unit tests for training and metric evaluation."""

import pandas as pd
from src.features import engineer_features, build_preprocessor
from src.evaluate import compute_metrics
from xgboost import XGBClassifier


def test_model_training_and_evaluation():
    df = pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male", "Female"] * 10,
            "SeniorCitizen": [0, 1, 0, 0, 1] * 10,
            "Partner": ["Yes", "No", "Yes", "No", "Yes"] * 10,
            "Dependents": ["No", "No", "Yes", "No", "No"] * 10,
            "tenure": [1, 24, 60, 12, 6] * 10,
            "PhoneService": ["Yes", "Yes", "Yes", "Yes", "No"] * 10,
            "MultipleLines": ["No", "Yes", "Yes", "No", "No phone service"] * 10,
            "InternetService": ["DSL", "Fiber optic", "Fiber optic", "DSL", "No"] * 10,
            "OnlineSecurity": ["No", "Yes", "Yes", "No", "No internet service"] * 10,
            "OnlineBackup": ["Yes", "No", "Yes", "No", "No internet service"] * 10,
            "DeviceProtection": ["No", "Yes", "Yes", "No", "No internet service"] * 10,
            "TechSupport": ["No", "Yes", "Yes", "No", "No internet service"] * 10,
            "StreamingTV": ["No", "Yes", "Yes", "No", "No internet service"] * 10,
            "StreamingMovies": ["No", "Yes", "Yes", "No", "No internet service"] * 10,
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
                "Month-to-month",
            ]
            * 10,
            "PaperlessBilling": ["Yes", "No", "Yes", "Yes", "No"] * 10,
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Electronic check",
                "Mailed check",
            ]
            * 10,
            "MonthlyCharges": [29.85, 89.85, 105.5, 55.0, 20.0] * 10,
            "TotalCharges": [29.85, 2156.4, 6330.0, 660.0, 120.0] * 10,
            "Churn": [1, 0, 0, 1, 0] * 10,
        }
    )

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_eng = engineer_features(X)
    preprocessor = build_preprocessor()
    X_trans = preprocessor.fit_transform(X_eng)

    model = XGBClassifier(n_estimators=10, random_state=42)
    model.fit(X_trans, y)

    metrics = compute_metrics(model, X_trans, y)
    assert "f1" in metrics
    assert "auc_roc" in metrics
    assert metrics["auc_roc"] >= 0.5
