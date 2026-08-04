"""
Feature engineering and preprocessing pipeline.
The same preprocessor is used in training AND inference — no leakage.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import logging

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "services_count",
    "charges_per_month_ratio",
    "streaming_bundle_score",
    "security_protection_score",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "SeniorCitizen",
    "tenure_group",
    "is_month_to_month",
]

SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Call before preprocessor.fit_transform()."""
    df = df.copy()
    # Tenure group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-np.inf, 12, 24, 48, 72, np.inf],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr", "6+yr"],
    ).astype(str)

    # Count active services
    df["services_count"] = (
        df[SERVICE_COLS]
        .apply(lambda col: col.isin(["Yes", "DSL", "Fiber optic"]))
        .sum(axis=1)
    )

    # Streaming bundle score
    streaming_cols = ["StreamingTV", "StreamingMovies"]
    df["streaming_bundle_score"] = (
        df[streaming_cols].apply(lambda col: col == "Yes").sum(axis=1)
    )

    # Security & Protection score
    security_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["security_protection_score"] = (
        df[security_cols].apply(lambda col: col == "Yes").sum(axis=1)
    )

    # High-vulnerability contract indicator
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(str)

    # Charges ratio — monthly spend relative to tenure
    df["charges_per_month_ratio"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Build sklearn ColumnTransformer. Call .fit() on training data only."""
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def save_preprocessor(preprocessor: ColumnTransformer, path: str) -> None:
    joblib.dump(preprocessor, path)
    logger.info(f"Saved preprocessor to {path}")


def load_preprocessor(path: str) -> ColumnTransformer:
    return joblib.load(path)
