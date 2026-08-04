"""
Evaluation utilities. Used in training and CI/CD performance gate.
"""

import sys
import numpy as np
import pandas as pd
import shap
import json
import joblib
from pathlib import Path
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    average_precision_score,
    roc_curve,
)
from typing import List, Dict, Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import engineer_features  # noqa: E402


def compute_fairness_metrics(model, X_raw: pd.DataFrame, y_true: pd.Series) -> Dict[str, Any]:
    """
    Compute per-subgroup fairness and bias metrics across sensitive attributes:
    - SeniorCitizen (Senior vs Non-Senior)
    - Contract (Month-to-month vs One year vs Two year)
    - TenureBucket (New 0-12m, Mid 13-36m, Long >36m)
    Returns selection rate, recall (TPR), FPR, F1 per subgroup, and disparity metrics.
    """
    df = X_raw.copy()
    X_engineered = engineer_features(df)
    y_pred = model.predict(X_engineered)

    subgroup_metrics = {}

    def get_group_stats(mask):
        sub_y = y_true[mask]
        sub_pred = y_pred[mask]
        if len(sub_y) == 0:
            return {"selection_rate": 0.0, "recall_tpr": 0.0, "fpr": 0.0, "f1": 0.0, "count": 0}
        
        selection_rate = float(np.mean(sub_pred))
        f1 = float(f1_score(sub_y, sub_pred, zero_division=0))
        rec = float(recall_score(sub_y, sub_pred, zero_division=0))
        
        neg_mask = (sub_y == 0)
        fpr = float(np.mean(sub_pred[neg_mask])) if np.sum(neg_mask) > 0 else 0.0
        
        return {
            "selection_rate": round(selection_rate, 4),
            "recall_tpr": round(rec, 4),
            "fpr": round(fpr, 4),
            "f1": round(f1, 4),
            "count": int(len(sub_y)),
        }

    # 1. SeniorCitizen
    if "SeniorCitizen" in df.columns:
        sen_mask = (df["SeniorCitizen"] == 1) | (df["SeniorCitizen"].astype(str).str.lower().isin(["1", "yes", "true"]))
        subgroup_metrics["SeniorCitizen"] = {
            "Senior": get_group_stats(sen_mask),
            "Non-Senior": get_group_stats(~sen_mask),
        }

    # 2. Contract
    if "Contract" in df.columns:
        subgroup_metrics["Contract"] = {}
        for ctype in ["Month-to-month", "One year", "Two year"]:
            c_mask = (df["Contract"].astype(str) == ctype)
            if np.sum(c_mask) > 0:
                subgroup_metrics["Contract"][ctype] = get_group_stats(c_mask)

    # 3. Tenure Bucket
    if "tenure" in df.columns:
        ten = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
        new_mask = (ten <= 12)
        mid_mask = (ten > 12) & (ten <= 36)
        long_mask = (ten > 36)
        subgroup_metrics["TenureBucket"] = {
            "New (0-12m)": get_group_stats(new_mask),
            "Mid (13-36m)": get_group_stats(mid_mask),
            "Long (>36m)": get_group_stats(long_mask),
        }

    # Disparities
    dp_diff = 0.0
    eo_diff = 0.0
    four_fifths_ratio = 1.0
    if "SeniorCitizen" in subgroup_metrics:
        s_stats = subgroup_metrics["SeniorCitizen"]["Senior"]
        ns_stats = subgroup_metrics["SeniorCitizen"]["Non-Senior"]
        sr_s = s_stats["selection_rate"]
        sr_ns = ns_stats["selection_rate"]
        dp_diff = round(abs(sr_s - sr_ns), 4)
        tpr_diff = abs(s_stats["recall_tpr"] - ns_stats["recall_tpr"])
        fpr_diff = abs(s_stats["fpr"] - ns_stats["fpr"])
        eo_diff = round(max(tpr_diff, fpr_diff), 4)
        
        # EEOC Four-Fifths Rule Ratio calculation: min_rate / max_rate >= 0.80
        max_sr = max(sr_s, sr_ns)
        min_sr = min(sr_s, sr_ns)
        four_fifths_ratio = round(min_sr / max_sr, 4) if max_sr > 0 else 1.0

    passes_four_fifths = (four_fifths_ratio >= 0.80)
    passes_flat_diff = (dp_diff <= 0.15) and (eo_diff <= 0.15)

    return {
        "subgroups": subgroup_metrics,
        "demographic_parity_difference": dp_diff,
        "equalized_odds_difference": eo_diff,
        "four_fifths_selection_ratio": four_fifths_ratio,
        "bias_status": "acceptable" if (passes_four_fifths and passes_flat_diff) else "disparity_detected",
    }


def compute_metrics(model, X: pd.DataFrame, y: pd.Series, X_raw: pd.DataFrame = None) -> Dict[str, Any]:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    res = {
        "f1": round(f1_score(y, y_pred), 4),
        "auc_roc": round(roc_auc_score(y, y_proba), 4),
        "auc_pr": round(average_precision_score(y, y_proba), 4),
        "precision": round(precision_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
    }
    if X_raw is not None:
        try:
            res["fairness"] = compute_fairness_metrics(model, X_raw, y)
        except Exception:
            pass
    return res


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
    model, preprocessor, X_raw: pd.DataFrame, n: int = 3
) -> List[Dict[str, Any]]:
    """
    Get top N SHAP factors for a single prediction row.
    Returns list of dicts with feature, value, impact, direction.
    """
    X_transformed = preprocessor.transform(X_raw)
    feature_names = get_feature_names(preprocessor)
    xgb_model = model.named_steps.get("model") if hasattr(model, "named_steps") else model

    sv = None

    # Stage 1: Try TreeExplainer
    try:
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = shap_values
    except Exception:
        pass

    # Stage 2: Fallback to model-agnostic / predict_proba Explainer
    if sv is None:
        try:
            explainer = shap.Explainer(xgb_model.predict_proba, X_transformed)
            res = explainer(X_transformed)
            sv = res.values[..., 1][0] if res.values.ndim == 3 else res.values[0]
        except Exception:
            pass

    # Stage 3: Fallback to feature importances
    if sv is None or len(sv) == 0:
        try:
            importances = getattr(xgb_model, "feature_importances_", None)
            if importances is not None and len(importances) == len(feature_names):
                sv = importances
        except Exception:
            pass

    if sv is None or len(sv) == 0:
        return []

    sorted_idx = np.argsort(np.abs(sv))[::-1]
    factors = []
    seen_features = set()

    for idx in sorted_idx:
        fname = feature_names[idx]
        orig_name = fname
        orig_val = "N/A"

        if fname in X_raw.columns:
            orig_name = fname
            orig_val = X_raw.iloc[0][fname]
        else:
            for col in X_raw.columns:
                if fname.startswith(col + "_"):
                    orig_name = col
                    orig_val = X_raw.iloc[0][col]
                    break

        if orig_name in seen_features:
            continue
        seen_features.add(orig_name)

        impact = float(sv[idx])
        factors.append(
            {
                "feature": orig_name,
                "value": str(orig_val),
                "impact": f"{impact:+.3f}",
                "direction": "increases_risk" if impact > 0 else "decreases_risk",
            }
        )
        if len(factors) == n:
            break
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
        "roc": [{"fpr": round(f, 4), "tpr": round(t, 4)} for f, t in zip(fpr, tpr)]
    }
    with open("reports/roc_curve.json", "w") as f:
        json.dump(roc_data, f, indent=2)

    print(f"Test metrics: {metrics}")
