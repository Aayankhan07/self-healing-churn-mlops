import os
import json
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Optional MLflow integration
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def evaluate():
    test_path = os.path.join("data", "processed", "test.csv")
    model_path = os.path.join("data", "model.pkl")

    if not os.path.exists(test_path) or not os.path.exists(model_path):
        raise FileNotFoundError("Model or test dataset not found. Please ensure data prep and training stages are run.")

    # Load data and pipeline
    print("Loading test dataset and pipeline model...")
    pipeline = joblib.load(model_path)
    df = pd.read_csv(test_path)

    X = df.drop(columns=["target"])
    y_true = df["target"]

    # Predict classifications and probabilities
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]  # Probabilities of class 1 (churn)

    # Compute classification metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    roc_auc = roc_auc_score(y_true, y_prob)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc)
    }

    # Save metrics to local JSON
    metrics_path = os.path.join("data", "metrics.json")
    os.makedirs("data", exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation complete. Metrics saved locally to {metrics_path}:")
    print(json.dumps(metrics, indent=2))

    # Log metrics to MLflow
    if MLFLOW_AVAILABLE:
        try:
            # Check if there is an active run. If not, start one.
            # In DVC flow, stages run as separate processes, so we start a run or log to the latest active run
            mlflow.set_experiment("customer-churn-prediction")
            with mlflow.start_run(run_name="RandomForest_Evaluation", nested=True):
                mlflow.log_metrics(metrics)
                print("Metrics logged successfully to MLflow tracking server.")
        except Exception as e:
            print(f"Could not log metrics to MLflow: {e}")

if __name__ == "__main__":
    evaluate()
