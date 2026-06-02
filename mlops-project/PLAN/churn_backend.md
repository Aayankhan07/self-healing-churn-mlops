# Backend Scheme
## ChurnGuard — Customer Churn Prediction Platform

**Version:** 1.0  
**Last Updated:** June 2026

---

## 1. Service Map

```
┌──────────────────────────────────────────────────────────────┐
│                        docker-compose                         │
│                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Streamlit  │    │    FastAPI      │    │   MLflow    │  │
│  │  :8501      │───▶│    :8000        │───▶│   :5000     │  │
│  └─────────────┘    │                 │    └─────────────┘  │
│                     │  ┌───────────┐  │                     │
│                     │  │  SQLite   │  │    ┌─────────────┐  │
│                     │  │  (local)  │  │    │    DVC      │  │
│                     │  └───────────┘  │    │  (remote)   │  │
│                     │                 │    └─────────────┘  │
│                     │  ┌───────────┐  │                     │
│                     │  │ Evidently │  │                     │
│                     │  │ (in-proc) │  │                     │
│                     │  └───────────┘  │                     │
│                     └─────────────────┘                     │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. API Layer — api/main.py

```python
# Startup sequence
app = FastAPI(title="ChurnGuard API")

@app.on_event("startup")
async def startup():
    # 1. Load model from MLflow registry (cached globally)
    model = mlflow.sklearn.load_model("models:/churn_model/Production")
    # 2. Load preprocessing pipeline
    pipeline = joblib.load("models/preprocessor.joblib")
    # 3. Init SQLite DB
    init_db()
    # 4. Load training reference data for Evidently
    reference_data = pd.read_csv("data/processed/train.csv")

# Routes registered:
# GET  /health
# GET  /metrics
# POST /predict          → predict.py:run_single_prediction()
# POST /predict/batch    → predict.py:run_batch_prediction()
# POST /upload           → predict.py:run_csv_prediction()
# GET  /drift/report     → drift.py:get_latest_report()
# GET  /drift/status     → drift.py:get_drift_status()
```

---

## 3. Prediction Logic — api/predict.py

```
run_single_prediction(input: CustomerInput) → PredictionOutput
    │
    ├── 1. Convert Pydantic model → pandas DataFrame (1 row)
    ├── 2. pipeline.transform(df) → scaled, encoded array
    ├── 3. model.predict_proba(array) → [p_no_churn, p_churn]
    ├── 4. p_churn = result[0][1]
    ├── 5. risk_tier = assign_tier(p_churn)
    ├── 6. shap_values = explainer.shap_values(array)
    ├── 7. top_3_factors = get_top_factors(shap_values, feature_names)
    ├── 8. prediction_id = uuid4()
    ├── 9. db.log_prediction(prediction_id, input_hash, p_churn, ...)
    └── 10. return PredictionOutput(...)
```

---

## 4. Pydantic Schemas — api/schemas.py

```python
# INPUT
class CustomerInput(BaseModel):
    tenure: int = Field(..., ge=0, description="Months as customer")
    MonthlyCharges: float = Field(..., gt=0)
    TotalCharges: float = Field(..., ge=0)
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaymentMethod: Literal["Electronic check", "Mailed check",
                           "Bank transfer (automatic)",
                           "Credit card (automatic)"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    # ... all 20 features with validation
    customerID: Optional[str] = None  # ignored in model

# OUTPUT
class ShapFactor(BaseModel):
    feature: str
    value: Any
    impact: str          # e.g. "+0.31"
    direction: Literal["increases_risk", "decreases_risk"]

class PredictionOutput(BaseModel):
    customer_id: Optional[str]
    churn_probability: float         # 0.0 – 1.0
    risk_tier: Literal["Low", "Medium", "High"]
    prediction: Literal[0, 1]
    top_factors: List[ShapFactor]    # top 3
    model_version: str
    prediction_id: str               # UUID
    timestamp: datetime
```

---

## 5. Database Layer — api/database.py

```python
# SQLAlchemy setup
engine = create_engine("sqlite:///./churnguard.db")

# Tables:
# predictions (id, customer_id, input_hash, probability,
#              risk_tier, prediction, model_ver, created_at)
# drift_reports (id, report_path, drift_detected,
#                drift_score, n_samples, created_at)

# Key queries used by dashboard:
# - count predictions last 30 days
# - group by risk_tier → distribution
# - daily prediction counts → trend chart
# - last 500 inputs → drift check
```

---

## 6. Monitoring Layer — api/drift.py

```
check_drift() called every 100 predictions
    │
    ├── Load last 500 prediction inputs from DB
    ├── Load training reference data
    ├── Run Evidently DataDriftPreset
    ├── Save HTML report to /reports/
    ├── Log to drift_reports table
    └── Return drift_status: {detected: bool, score: float, tier: str}

get_drift_status() → reads latest row from drift_reports table
get_latest_report() → returns HTML file as FileResponse
```

---

## 7. ML Layer — src/

```
src/data_prep.py
    load_raw_data(path) → DataFrame
    clean_data(df) → DataFrame          # fix TotalCharges, drop nulls
    split_data(df) → train, val, test   # stratified on Churn

src/features.py
    build_preprocessor() → ColumnTransformer
        numeric_features → SimpleImputer + StandardScaler
        categorical_features → SimpleImputer + OneHotEncoder
    engineer_features(df) → DataFrame
        tenure_group, services_count, charges_ratio

src/train.py
    with mlflow.start_run():
        mlflow.log_params(params)
        pipeline = Pipeline([preprocessor, smote, model])
        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_val, y_val)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipeline, "model")
        mlflow.register_model(...)

src/evaluate.py
    evaluate(model, X, y) → {f1, auc, precision, recall}
    get_shap_values(model, X) → shap_values array
    get_top_factors(shap_values, feature_names, n=3) → List[ShapFactor]

src/monitor.py
    generate_drift_report(current_data, reference_data) → report_path
    compute_psi(reference, current, feature) → float
```

---

## 8. Training Pipeline — dvc.yaml

```yaml
stages:
  prepare:
    cmd: python src/data_prep.py
    deps: [data/raw/telco_churn.csv, src/data_prep.py]
    outs: [data/processed/train.csv, data/processed/val.csv,
           data/processed/test.csv]
    params: [params.yaml:data]

  train:
    cmd: python src/train.py
    deps: [data/processed/train.csv, src/train.py, src/features.py]
    outs: [models/preprocessor.joblib]
    params: [params.yaml:model]
    metrics: [metrics/train_metrics.json]

  evaluate:
    cmd: python src/evaluate.py
    deps: [data/processed/test.csv, src/evaluate.py]
    metrics: [metrics/eval_metrics.json]
    plots: [reports/confusion_matrix.json, reports/roc_curve.json]
```

---

## 9. Configuration — params.yaml

```yaml
data:
  test_size: 0.2
  val_size: 0.1
  random_seed: 42
  target_column: Churn

model:
  algorithm: xgboost
  n_estimators: 300
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  scale_pos_weight: 2.7    # handles class imbalance
  eval_metric: aucpr

thresholds:
  risk_low: 0.35
  risk_high: 0.65
  min_f1: 0.80
  drift_mild: 0.10
  drift_significant: 0.20

api:
  batch_max_rows: 5000
  upload_max_mb: 10
  drift_check_every_n: 100
```

---

## 10. Docker Setup

```dockerfile
# Dockerfile (API)
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY api/ ./api/
COPY models/ ./models/
COPY data/processed/ ./data/processed/
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  api:
    build: .
    ports: ["8000:8000"]
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - DATABASE_URL=sqlite:///./churnguard.db
    volumes:
      - ./reports:/app/reports
      - ./churnguard.db:/app/churnguard.db
    depends_on: [mlflow]

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
    command: mlflow server --host 0.0.0.0
    volumes:
      - ./mlruns:/mlruns

  dashboard:
    image: python:3.10-slim
    ports: ["8501:8501"]
    command: streamlit run dashboard/app.py
    environment:
      - API_URL=http://api:8000
    depends_on: [api]
```
