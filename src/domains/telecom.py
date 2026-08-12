"""
Telecom domain specification.

This is the original ChurnGuard schema, extracted verbatim from the healing
rules that used to live in api/main.py. The action strings are part of the
observable API contract (they are returned in `healed_actions` and written to
the self-healing log), so they are reproduced exactly.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from src.features import engineer_features

from .base import (
    BinaryRule,
    CategoricalRule,
    Constraint,
    DomainSpec,
    NumericRule,
    RiskBands,
    round_derived,
)

# Median monthly charge of the training population, used to fill a missing value.
MEDIAN_MONTHLY_CHARGES = 70.35


def _derive_total_charges(row: Dict[str, Any]) -> float:
    """TotalCharges is a function of the monthly rate and the tenure so far."""
    return round_derived(row["MonthlyCharges"] * max(1, row["tenure"]))


NUMERIC_RULES = [
    NumericRule(
        name="tenure",
        cast=int,
        minimum=0,
        default=0,
        label_missing="Imputed missing tenure to 0",
        label_coerced="Coerced tenure to integer",
        label_invalid="Imputed invalid tenure to 0",
        label_clamped="Clamped negative tenure to 0",
    ),
    NumericRule(
        name="MonthlyCharges",
        cast=float,
        minimum=0.01,
        minimum_exclusive=True,
        default=MEDIAN_MONTHLY_CHARGES,
        label_missing=(
            f"Imputed missing MonthlyCharges to median ({MEDIAN_MONTHLY_CHARGES})"
        ),
        label_coerced="Coerced MonthlyCharges to float",
        label_invalid=(
            f"Imputed invalid MonthlyCharges to median ({MEDIAN_MONTHLY_CHARGES})"
        ),
        label_clamped="Clamped non-positive MonthlyCharges to 0.01",
    ),
    NumericRule(
        name="TotalCharges",
        cast=float,
        minimum=0.0,
        derive=_derive_total_charges,
        label_missing="Imputed missing TotalCharges as MonthlyCharges * tenure",
        label_coerced="Coerced TotalCharges to float",
        label_invalid="Imputed invalid TotalCharges as MonthlyCharges * tenure",
        label_clamped="Clamped negative TotalCharges to 0.0",
    ),
]

BINARY_RULES = [BinaryRule(name="SeniorCitizen")]

CATEGORICAL_RULES = [
    CategoricalRule("gender", ["Male", "Female"]),
    CategoricalRule("Partner", ["Yes", "No"]),
    CategoricalRule("Dependents", ["Yes", "No"]),
    CategoricalRule("PhoneService", ["Yes", "No"]),
    CategoricalRule("MultipleLines", ["Yes", "No", "No phone service"]),
    CategoricalRule("InternetService", ["DSL", "Fiber optic", "No"]),
    CategoricalRule("OnlineSecurity", ["Yes", "No", "No internet service"]),
    CategoricalRule("OnlineBackup", ["Yes", "No", "No internet service"]),
    CategoricalRule("DeviceProtection", ["Yes", "No", "No internet service"]),
    CategoricalRule("TechSupport", ["Yes", "No", "No internet service"]),
    CategoricalRule("StreamingTV", ["Yes", "No", "No internet service"]),
    CategoricalRule("StreamingMovies", ["Yes", "No", "No internet service"]),
    CategoricalRule("Contract", ["Month-to-month", "One year", "Two year"]),
    CategoricalRule("PaperlessBilling", ["Yes", "No"]),
    CategoricalRule(
        "PaymentMethod",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    ),
]


def _tenure_positive(row: Dict[str, Any]) -> bool:
    return row.get("tenure", 0) > 0


def _total_covers_monthly(row: Dict[str, Any]) -> bool:
    return row["TotalCharges"] >= row["MonthlyCharges"]


def _recompute_total(row: Dict[str, Any]) -> Dict[str, Any]:
    row["TotalCharges"] = round_derived(row["MonthlyCharges"] * row["tenure"])
    return row


CONSTRAINTS = [
    Constraint(
        name="total_charges_covers_monthly",
        applies=_tenure_positive,
        holds=_total_covers_monthly,
        repair=_recompute_total,
        message=(
            "Recomputed TotalCharges as MonthlyCharges * tenure "
            "due to constraint mismatch"
        ),
    ),
]

TELECOM_SPEC = DomainSpec(
    key="telecom",
    display_name="Telecom Customer Churn",
    numeric=NUMERIC_RULES,
    binary=BINARY_RULES,
    categorical=CATEGORICAL_RULES,
    constraints=CONSTRAINTS,
    risk_bands=RiskBands(low=0.35, high=0.65),
    target_column="Churn",
    id_column="customerID",
    label_source_path=os.path.join("data", "raw", "telco_churn.csv"),
    feature_engineering=engineer_features,
)
