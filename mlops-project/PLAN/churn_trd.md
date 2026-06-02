# Technical Requirements Document (TRD)
## ChurnGuard — Customer Churn Prediction Platform

**Version:** 1.0  
**Author:** Solo Developer  
**Status:** Draft  
**Last Updated:** June 2026

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLIENT LAYER                         │
│         Streamlit Dashboard (browser)                    │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP
┌────────────────────▼────────────────────────────────────┐
│                     API LAYER                            │
│         FastAPI (uvicorn) — Port 8000                    │
│   /health  /predict  /predict/batch  /metrics  /drift   │
└────────────────────┬────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
┌───────▼──────┐        ┌────────▼────────┐
│  ML LAYER    │        │  STORAGE LAYER  │
│  MLflow      │        │  SQLite DB      │
│  Model       │        │  (predictions)  │
│  Registry    │        │                 │
│  SHAP        │        │  DVC Remote     │
│  Evidently   │        │  (data/models)  │
└──────────────┘        └─────────────────┘
```

---

## 2. Technology Stack

### 2.1 ML & Data
| Component | Technology | Version | Reason |
|---|---|---|---|
| ML Framework | Scikit-learn | ≥ 1.4 | Pipeline + preprocessing |
| Gradient Boosting | XGBoost | ≥ 2.0 | Best accuracy on tabular data |
| Hyperparameter tuning | Optuna | ≥ 3.5 | Efficient TPE search |
| Experiment tracking | MLflow | ≥ 2.11 | Model registry + run comparison |
| Explainability | SHAP | ≥ 0.45 | TreeExplainer for XGBoost |
| Imbalance handling | imbalanced-learn | ≥ 0.12 | SMOTE |
| Data versioning | DVC | ≥ 3.0 | Git-like data tracking |
| Drift monitoring | Evidently AI | ≥ 0.4 | Data drift + model monitoring |

### 2.2 API & Backend
| Component | Technology | Version | Reason |
|---|---|---|---|
| API framework | FastAPI | ≥ 0.111 | Async, auto Swagger docs, Pydantic |
| ASGI server | Uvicorn | ≥ 0.29 | Production-grade async server |
| Data validation | Pydantic v2 | ≥ 2.7 | Strict input schema enforcement |
| Database | SQLite + SQLAlchemy | ≥ 2.0 | Lightweight, no setup, sufficient for portfolio |
| CSV handling | Pandas | ≥ 2.2 | File parsing + batch processing |

### 2.3 Frontend
| Component | Technology | Reason |
|---|---|---|
| Dashboard | Streamlit | Fast to build, easy to deploy, good for data apps |
| Charts | Plotly | Interactive, beautiful, Streamlit-native |
| SHAP plots | SHAP + Matplotlib | Waterfall charts for explanations |

### 2.4 DevOps & Infrastructure
| Component | Technology | Reason |
|---|---|---|
| Containerization | Docker + docker-compose | Reproducible, one-command startup |
| CI/CD | GitHub Actions | Free, integrates with repo |
| Code quality | flake8 + black | Linting + formatting |
| Testing | pytest + pytest-cov | Unit + integration tests |
| API deployment | Render (free tier) | Simple Docker deployment, free |
| Dashboard deployment | Streamlit Cloud (free) | One-click deploy from GitHub |
| Model/data remote | DagsHub | Free MLflow + DVC remote |

---

## 3. Data Schema

### 3.1 Input Features (20 features)
| Feature | Type | Description | Validation |
|---|---|---|---|
| tenure | int | Months as customer | ≥ 0 |
| MonthlyCharges | float | Monthly bill amount | > 0 |
| TotalCharges | float | Total billed to date | ≥ 0 |
| Contract | str | Month-to-month / One year / Two year | enum |
| PaymentMethod | str | Electronic check / Mail check / Bank transfer / Credit card | enum |
| InternetService | str | DSL / Fiber optic / No | enum |
| OnlineSecurity | str | Yes / No / No internet service | enum |
| OnlineBackup | str | Yes / No / No internet service | enum |
| TechSupport | str | Yes / No / No internet service | enum |
| StreamingTV | str | Yes / No / No internet service | enum |
| StreamingMovies | str | Yes / No / No internet service | enum |
| PhoneService | str | Yes / No | enum |
| MultipleLines | str | Yes / No / No phone service | enum |
| PaperlessBilling | str | Yes / No | enum |
| gender | str | Male / Female | enum |
| SeniorCitizen | int | 0 or 1 | {0,1} |
| Partner | str | Yes / No | enum |
| Dependents | str | Yes / No | enum |
| DeviceProtection | str | Yes / No / No internet service | enum |
| customerID | str | Unique identifier | optional, not used in model |

### 3.2 Output Schema
```json
{
  "customer_id": "string | null",
  "churn_probability": 0.73,
  "risk_tier": "High",
  "prediction": 1,
  "top_factors": [
    {"feature": "Contract", "value": "Month-to-month", "impact": "+0.31", "direction": "increases_risk"},
    {"feature": "tenure", "value": 3, "impact": "+0.18", "direction": "increases_risk"},
    {"feature": "OnlineSecurity", "value": "No", "impact": "+0.12", "direction": "increases_risk"}
  ],
  "model_version": "1.0.2",
  "prediction_id": "uuid",
  "timestamp": "2026-06-02T10:30:00Z"
}
```

### 3.3 Risk Tier Thresholds
| Tier | Probability Range | Color |
|---|---|---|
| Low | 0.00 – 0.35 | Green |
| Medium | 0.35 – 0.65 | Amber |
| High | 0.65 – 1.00 | Red |

---

## 4. API Specification

### 4.1 Endpoints
| Method | Path | Description | Auth |
|---|---|---|---|
| GET | /health | Liveness check | None |
| GET | /metrics | Model version, uptime, prediction count | None |
| POST | /predict | Single customer prediction | API Key (header) |
| POST | /predict/batch | Batch predictions from JSON array | API Key (header) |
| POST | /upload | CSV file upload → predictions CSV | API Key (header) |
| GET | /drift/report | Latest Evidently drift report (HTML) | None |
| GET | /drift/status | Drift status summary (JSON) | None |

### 4.2 Rate Limits
| Endpoint | Limit |
|---|---|
| /predict | 100 req/min |
| /predict/batch | 10 req/min |
| /upload | 5 req/min |

### 4.3 Error Codes
| Code | Meaning |
|---|---|
| 200 | Success |
| 422 | Validation error — invalid input fields |
| 429 | Rate limit exceeded |
| 500 | Internal server error — model inference failed |
| 503 | Model not loaded |

---

## 5. ML Pipeline Specification

### 5.1 Preprocessing Steps (in order)
1. Drop `customerID` column
2. Convert `TotalCharges` to numeric (handle empty strings → NaN)
3. Impute missing `TotalCharges` with median
4. Binary encode: `gender`, `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`
5. One-hot encode: `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaymentMethod`
6. Scale: `tenure`, `MonthlyCharges`, `TotalCharges` with StandardScaler
7. Engineer features: `tenure_group` (bins), `services_count` (sum of active services), `charges_per_month_ratio`
8. SMOTE oversampling on training set only (never on test set)

### 5.2 Model Selection Criteria
- Primary metric: F1 Score (churn class) — recall of actual churners matters most
- Secondary: AUC-ROC, Precision-Recall AUC
- Constraint: inference time < 50ms per prediction
- Winner: XGBoost with Optuna-tuned hyperparameters (expected F1 ~0.83)

### 5.3 Retraining Triggers
| Trigger | Condition |
|---|---|
| Scheduled | Monthly |
| Drift-based | PSI > 0.2 on any top-5 feature |
| Performance-based | F1 drops > 5% on validation set |

---

## 6. Database Schema

### predictions table
```sql
CREATE TABLE predictions (
    id          TEXT PRIMARY KEY,      -- UUID
    customer_id TEXT,                  -- optional
    input_hash  TEXT NOT NULL,         -- SHA256 of input (no PII in logs)
    probability REAL NOT NULL,
    risk_tier   TEXT NOT NULL,
    prediction  INTEGER NOT NULL,
    model_ver   TEXT NOT NULL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### drift_reports table
```sql
CREATE TABLE drift_reports (
    id              TEXT PRIMARY KEY,
    report_path     TEXT NOT NULL,
    drift_detected  INTEGER NOT NULL,   -- 0 or 1
    drift_score     REAL,
    n_samples       INTEGER,
    created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 7. File Structure

```
churnguard/
├── data/
│   ├── raw/                    # DVC tracked
│   │   └── telco_churn.csv
│   └── processed/              # DVC tracked
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── src/
│   ├── data_prep.py            # Load, clean, split data
│   ├── features.py             # Feature engineering functions
│   ├── train.py                # Training script (MLflow logging)
│   ├── evaluate.py             # Evaluation metrics + SHAP
│   └── monitor.py              # Evidently drift reports
├── api/
│   ├── main.py                 # FastAPI app + routes
│   ├── schemas.py              # Pydantic input/output models
│   ├── predict.py              # Inference logic
│   ├── database.py             # SQLAlchemy setup + queries
│   └── drift.py                # Drift check on predictions
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_pipeline.py
│   ├── test_performance.py
│   └── test_api.py
├── .github/
│   └── workflows/
│       └── ci.yml
├── notebooks/
│   └── eda.ipynb               # Exploratory analysis (not production)
├── Dockerfile
├── docker-compose.yml
├── dvc.yaml                    # Pipeline stages
├── params.yaml                 # Hyperparameters + thresholds
├── requirements.txt
├── .dvcignore
├── .gitignore
└── README.md
```

---

## 8. CI/CD Pipeline

```
Push to GitHub
     │
     ▼
GitHub Actions triggered
     │
     ├── Install dependencies
     ├── flake8 linting
     ├── black format check
     ├── pytest (unit + integration)
     ├── Coverage check (≥ 80%)
     ├── Model performance gate (F1 ≥ 0.80)
     │
     └── On merge to main:
         ├── docker build
         ├── docker push → Docker Hub
         └── Render auto-deploy (webhook)
```

---

## 9. Environment Variables

| Variable | Description | Required |
|---|---|---|
| MLFLOW_TRACKING_URI | DagsHub or local MLflow URI | Yes |
| MLFLOW_MODEL_NAME | Registered model name | Yes |
| MLFLOW_MODEL_STAGE | Production / Staging | Yes |
| API_KEY | Simple API key for /predict endpoints | Yes |
| DATABASE_URL | SQLite path or Postgres URI | Yes |
| DRIFT_THRESHOLD | PSI threshold for drift alert | Yes (default: 0.2) |
| DAGSHUB_TOKEN | DagsHub auth token | Yes (for remote) |
