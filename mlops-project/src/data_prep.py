"""
Load, clean, and split the Telco churn dataset.
Tracked by DVC — outputs go to data/processed/.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Fix TotalCharges — contains spaces, should be float
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Fill nulls in TotalCharges with median
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    # Convert Churn to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    # Drop customerID — not a feature
    df = df.drop(columns=["customerID"], errors="ignore")
    logger.info(f"Cleaned data. Shape: {df.shape}. Nulls: {df.isnull().sum().sum()}")
    return df


def split_data(df: pd.DataFrame, params: dict) -> tuple:
    target = params["target_column"]
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=params["test_size"] + params["val_size"],
        random_state=params["random_seed"],
        stratify=y
    )
    val_ratio = params["val_size"] / (params["test_size"] + params["val_size"])
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_ratio,
        random_state=params["random_seed"],
        stratify=y_temp
    )
    logger.info(f"Split: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["data"]

    df = load_raw_data("data/raw/telco_churn.csv")
    df = clean_data(df)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, params)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    X_train.assign(Churn=y_train).to_csv("data/processed/train.csv", index=False)
    X_val.assign(Churn=y_val).to_csv("data/processed/val.csv", index=False)
    X_test.assign(Churn=y_test).to_csv("data/processed/test.csv", index=False)
    logger.info("Saved processed splits to data/processed/")
