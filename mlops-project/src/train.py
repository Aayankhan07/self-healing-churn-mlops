import os
import yaml
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Optional MLflow integration
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

def load_params(params_path="params.yaml"):
    with open(params_path, "r") as f:
        return yaml.safe_load(f)

def train():
    params = load_params()
    
    # Extract configs
    random_state = params["base"]["random_state"]
    num_cols = params["data_prep"]["numerical_cols"]
    cat_cols = params["data_prep"]["categorical_cols"]
    
    n_estimators = params["train"]["n_estimators"]
    max_depth = params["train"]["max_depth"]
    min_samples_split = params["train"]["min_samples_split"]
    class_weight = params["train"]["class_weight"]

    # Load datasets
    train_path = os.path.join("data", "processed", "train.csv")
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Prepared train data not found at {train_path}. Please run data prep stage first.")

    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    X = df.drop(columns=["target"])
    y = df["target"]

    # Create Scikit-learn Pipeline
    print("Constructing pre-processing pipeline and column transformers...")
    numerical_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ]
    )

    # Combine preprocessor with RandomForest classifier
    full_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight=class_weight,
            random_state=random_state
        ))
    ])

    # Optionally log training parameters with MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_experiment("customer-churn-prediction")
            # If there's an active run, we can log to it, or start a new one
            mlflow.start_run(run_name="RandomForest_Train")
            
            # Log params
            mlflow.log_params({
                "estimator": "RandomForestClassifier",
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "class_weight": class_weight,
                "random_state": random_state
            })
            print("MLflow experiment run started and parameters logged.")
        except Exception as e:
            print(f"MLflow tracking started with errors (falling back to standard local run): {e}")

    # Train model pipeline
    print("Fitting Scikit-learn Pipeline...")
    full_pipeline.fit(X, y)

    # Save model artifact locally (required by DVC)
    os.makedirs("data", exist_ok=True)
    model_path = os.path.join("data", "model.pkl")
    joblib.dump(full_pipeline, model_path)
    print(f"Pipeline successfully trained and saved locally to {model_path}")

    # Log model artifact with MLflow
    if MLFLOW_AVAILABLE:
        try:
            mlflow.sklearn.log_model(full_pipeline, "model")
            mlflow.end_run()
            print("Model logged to MLflow Model Registry.")
        except Exception as e:
            print(f"Could not log model artifact to MLflow: {e}")

if __name__ == "__main__":
    train()
