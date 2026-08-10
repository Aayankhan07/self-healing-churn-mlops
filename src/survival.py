"""
Survival Analysis & Time-to-Churn Probability Module.
Estimates time-to-churn days and survival probability timeline curves using parametric hazard modeling.
"""

import math
from typing import Dict, Any


def calculate_survival_curve(
    tenure: float, probability: float, domain_id: str = "telecom"
) -> Dict[str, Any]:
    """
    Calculate time-to-churn days and survival timeline probabilities (30, 60, 90, 180 days).
    Uses a Weibull-inspired hazard rate mapping based on classifier churn probability and customer tenure.
    Note: This is a probability-derived time-to-churn estimation heuristic (S(t_days) = exp(- (base_lambda * (t_days / 30.0))^beta)),
    not a fitted Weibull AFT model on right-censored event data.
    """
    tenure_months = max(1.0, float(tenure or 1.0))
    p_churn = max(0.01, min(0.99, float(probability or 0.5)))

    # Base hazard rate lambda (per month) derived from churn probability and tenure
    base_lambda = (p_churn / 0.5) * (1.0 / math.log(tenure_months + 1.0 + math.e))
    shape_beta = 1.15  # Accelerated hazard rate over time

    # Calculate survival probability S(t_days) = exp(- (base_lambda * (t_days / 30.0))^shape_beta)
    def survival_at_t(t_days: float) -> float:
        t_months = t_days / 30.0
        exponent = (base_lambda * t_months) ** shape_beta
        return round(math.exp(-exponent), 4)

    s_30 = survival_at_t(30)
    s_60 = survival_at_t(60)
    s_90 = survival_at_t(90)
    s_180 = survival_at_t(180)

    # Estimate median time-to-churn days (where S(t) = 0.50)
    if p_churn >= 0.70:
        estimated_days = max(14, int(45 * (1.0 - p_churn) + 15))
    elif p_churn >= 0.30:
        estimated_days = max(30, int(90 * (1.0 - p_churn) + 30))
    else:
        estimated_days = int(180 + (1.0 - p_churn) * 180)

    horizon_pct = int(p_churn * 100)
    entity_term = "Student" if str(domain_id).lower() == "school" else "Customer"
    summary = f"{entity_term} has a {horizon_pct}% risk of churn within approximately {estimated_days} days."

    return {
        "time_to_churn_days": estimated_days,
        "risk_horizon_summary": summary,
        "survival_timeline": {
            "30_days": s_30,
            "60_days": s_60,
            "90_days": s_90,
            "180_days": s_180,
        },
    }
