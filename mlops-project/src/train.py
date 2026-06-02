"""
Training script. Run directly or via DVC.
Logs everything to MLflow. Registers best model in MLflow registry.
"""
import pandas as pd
import numpy as np
import yaml
import mlflow
import mlflow.sklearn
import optuna
import logging
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.features import engineer_features, build_preprocessor, save_preprocessor
from src.evaluate import compute_metrics, get_feature_names

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_split(path: str, target: str = "Churn"):
    df = pd.read_csv(path)
    return df.drop(columns=[target]), df[target]


def train(params: dict):
    tracking_uri = params.get("mlflow_uri", "http://localhost:5000")
    import socket
    try:
        with socket.create_connection(("localhost", 5000), timeout=0.5):
            pass
    except Exception:
        tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("churnguard-churn-prediction")

    X_train_raw, y_train = load_split("data/processed/train.csv")
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
        pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(**model_params))
        ])
        pipeline.fit(X_train, y_train)
        metrics = compute_metrics(pipeline, X_val, y_val)
        return metrics["f1"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    best_params = study.best_params
    best_params.update({"eval_metric": "aucpr", "use_label_encoder": False, "random_state": 42})

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(best_params)

        final_pipeline = ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=42)),
            ("model", XGBClassifier(**best_params))
        ])
        final_pipeline.fit(X_train, y_train)

        metrics = compute_metrics(final_pipeline, X_val, y_val)
        mlflow.log_metrics(metrics)
        logger.info(f"Val metrics: {metrics}")

        # Save preprocessor separately for inference (needed for SHAP)
        Path("models").mkdir(exist_ok=True)
        save_preprocessor(preprocessor, "models/preprocessor.joblib")
        import joblib
        joblib.dump(final_pipeline, "models/model.joblib")
        mlflow.log_artifact("models/preprocessor.joblib")
        mlflow.log_artifact("models/model.joblib")

        # Log + register full pipeline
        mlflow.sklearn.log_model(final_pipeline, "model")
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, "churn_model")

        # Save metrics for DVC
        Path("metrics").mkdir(exist_ok=True)
        import json
        with open("metrics/eval_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info("Training complete. Model registered.")


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
