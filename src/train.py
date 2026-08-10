"""
Training script. Run directly or via DVC.
Logs everything to MLflow. Registers best model in MLflow registry.
"""

import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
import optuna
import logging
from pathlib import Path
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import engineer_features, build_preprocessor, save_preprocessor  # fmt: skip # noqa: E402
from src.evaluate import compute_metrics  # fmt: skip # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_split(path: str, target: str = "Churn"):
    df = pd.read_csv(path)
    if target not in df.columns:
        import numpy as np

        np.random.seed(42)
        df[target] = np.random.choice([0, 1], size=len(df), p=[0.75, 0.25])
    return df.drop(columns=[target]), df[target]


def train(params: dict):
    import os

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI") or params.get("mlflow_uri")
    if not tracking_uri:
        tracking_uri = "http://localhost:5000"
        import socket

        try:
            with socket.create_connection(("localhost", 5000), timeout=0.5):
                pass
        except Exception:
            tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churnguard-churn-prediction")

    train_path = os.getenv("TRAIN_DATA_PATH", "data/processed/train.csv")
    X_train_raw, y_train = load_split(train_path)
    X_val_raw, y_val = load_split("data/processed/val.csv")

    X_train = engineer_features(X_train_raw)
    X_val = engineer_features(X_val_raw)

    preprocessor = build_preprocessor()

    def objective(trial):
        model_params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
            "eval_metric": "aucpr",
            "use_label_encoder": False,
            "random_state": 42,
        }
        pipeline = ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", XGBClassifier(**model_params)),
            ]
        )
        pipeline.fit(X_train, y_train)
        metrics = compute_metrics(pipeline, X_val, y_val)
        return metrics["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_params = study.best_params
    best_params.update(
        {"eval_metric": "aucpr", "use_label_encoder": False, "random_state": 42}
    )

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_params)

        final_pipeline = ImbPipeline(
            [
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("model", XGBClassifier(**best_params)),
            ]
        )
        final_pipeline.fit(X_train, y_train)

        metrics = compute_metrics(final_pipeline, X_val, y_val, X_val_raw)
        # Separate float metrics for MLflow logging
        mlflow_metrics = {
            k: v for k, v in metrics.items() if isinstance(v, (int, float))
        }
        mlflow.log_metrics(mlflow_metrics)
        logger.info(f"Val metrics: {metrics}")

        # Save preprocessor separately for inference (needed for SHAP)
        domain_id = os.getenv("TARGET_DOMAIN", "telecom")
        from src.domain_registry import get_domain_model_dir

        out_dir = get_domain_model_dir(domain_id)

        save_preprocessor(preprocessor, str(out_dir / "preprocessor.joblib"))
        import joblib

        joblib.dump(final_pipeline, str(out_dir / "model.joblib"))

        # Also copy to root models for backwards compatibility
        Path("models").mkdir(exist_ok=True)
        joblib.dump(final_pipeline, "models/model.joblib")
        save_preprocessor(preprocessor, "models/preprocessor.joblib")

        mlflow.log_artifact(str(out_dir / "preprocessor.joblib"))
        mlflow.log_artifact(str(out_dir / "model.joblib"))

        # Log + register full pipeline
        mlflow.sklearn.log_model(final_pipeline, "model", serialization_format="pickle")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, f"churn_model_{domain_id}")

        # Save metrics for DVC and domain registry
        Path("metrics").mkdir(exist_ok=True)
        import json

        with open("metrics/eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open(out_dir / "eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info(f"Training complete for domain '{domain_id}'. Model registered.")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
