"""
Evaluation utilities. Used in training and CI/CD performance gate.
"""
import numpy as np
import pandas as pd
import shap
import json
import joblib
from pathlib import Path
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, average_precision_score, roc_curve
)
from typing import List, Dict, Any
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import engineer_features

def compute_metrics(model, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "f1": round(f1_score(y, y_pred), 4),
        "auc_roc": round(roc_auc_score(y, y_proba), 4),
        "auc_pr": round(average_precision_score(y, y_proba), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
    }


def get_feature_names(preprocessor) -> List[str]:
    """Extract feature names after ColumnTransformer."""
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(cols).tolist())
        else:
            names.extend(cols)
    return names


def get_top_shap_factors(
    model,
    preprocessor,
    X_raw: pd.DataFrame,
    n: int = 3
) -> List[Dict[str, Any]]:
    """
    Get top N SHAP factors for a single prediction row.
    Returns list of dicts with feature, value, impact, direction.
    """
    X_transformed = preprocessor.transform(X_raw)
    feature_names = get_feature_names(preprocessor)

    # Use TreeExplainer for XGBoost (fast)
    xgb_model = model.named_steps.get("model") or model
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_transformed)

    if shap_values.ndim == 1:
        sv = shap_values
    else:
        sv = shap_values[0]

    top_idx = np.argsort(np.abs(sv))[::-1][:n]
    factors = []
    for idx in top_idx:
        fname = feature_names[idx]
        # Map back to original feature name (strip OHE prefix)
        orig_name = fname.split("__")[-1].split("_")[0] if "__" in fname else fname
        orig_val = X_raw.iloc[0].get(orig_name, "N/A")
        impact = sv[idx]
        factors.append({
            "feature": orig_name,
            "value": str(orig_val),
            "impact": f"{impact:+.3f}",
            "direction": "increases_risk" if impact > 0 else "decreases_risk",
        })
    return factors


if __name__ == "__main__":
    # DVC evaluation stage entrypoint
    Path("metrics").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # Load test split
    test_df = pd.read_csv("data/processed/test.csv")
    X_test_raw = test_df.drop(columns=["Churn"])
    y_test = test_df["Churn"]

    X_test = engineer_features(X_test_raw)

    # Load pipeline
    pipeline = joblib.load("models/model.joblib")

    # Evaluate
    metrics = compute_metrics(pipeline, X_test, y_test)

    # Save metrics
    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate ROC curve plot data
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data = {
        "roc": [
            {"fpr": round(f, 4), "tpr": round(t, 4)}
            for f, t in zip(fpr, tpr)
        ]
    }
    with open("reports/roc_curve.json", "w") as f:
        json.dump(roc_data, f, indent=2)

    print(f"Test metrics: {metrics}")
