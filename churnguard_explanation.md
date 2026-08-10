# ChurnGuard: Deep Codebase Explanation & Architectural Guide

This document provides an exhaustive, production-grade review of the ChurnGuard multi-domain customer churn prediction codebase.

---

## Repository Structure

```text
├── api/                     # FastAPI backend microservice
│   ├── database.py          # SQLAlchemy SQLite models with domain_id schema tagging
│   ├── main.py              # Main API service entry point & domain routing
│   ├── metrics_prometheus.py# Prometheus exposition metrics endpoint
│   ├── predict.py           # Single/batch prediction & playbooks integration
│   └── schemas.py           # Pydantic v2 schemas for request validation
├── dashboard/               # Streamlit frontend service
│   └── app.py               # Dark-mode dashboard visualization & domain switcher
├── data/                    # Storage for datasets and per-domain baselines
├── dvc.yaml                 # DVC lifecycle stage declarations
├── metrics/                 # Performance indicators (JSON outputs)
├── mlflow.db                # SQLite database for MLflow tracking
├── models/                  # Domain-isolated model directories
│   ├── ecommerce/
│   ├── fitness/
│   ├── school/
│   └── telecom/
├── params.yaml              # Hyperparameter and operational configurations
├── reports/                 # Auto-generated Evidently HTML drift reports
├── scripts/                 # Execution scripts (e.g. train_all_domains.py)
├── src/                     # Raw Python pipelines
│   ├── domain_registry.py   # Domain artifact & baseline manager
│   ├── evaluate.py          # Model testing metrics & SHAP factors
│   ├── features.py          # Custom feature engineering pipeline
│   ├── monitor.py           # Evidently AI drift detector
│   ├── notifications.py     # Webhook alert dispatcher
│   ├── playbooks.py         # Retention action playbooks generator
│   ├── survival.py          # Parametric survival curve calculator
│   └── train.py             # Optuna optimization and training executor
└── tests/                   # Pytest testing suite (20/20 passing)
```

---

## Architectural Overview

ChurnGuard is an autonomous, self-healing MLOps system designed to predict customer and student churn across multiple industry domains (Telecom, K-12 Schools, E-Commerce, Fitness, and Custom domains).

### Key Architectural Pillars

1. **Domain-Isolated Multi-Model Registry (`src/domain_registry.py`)**:
   - Isolates model artifacts (`models/{domain_id}/model.joblib`), preprocessors, and baselines (`data/baselines/{domain_id}_baseline.csv`).
   - Routes inference requests dynamically based on the requested domain.

2. **Ingestion & Retraining Self-Healing**:
   - Rule-based data healing (clamping negative values, imputing missing totals, categorical string similarity matching) cleans input payloads automatically before prediction and retraining.

3. **Automated Retention Action Playbooks (`src/playbooks.py`)**:
   - Maps SHAP drivers and customer features into domain-tailored intervention playbooks.

4. **Survival Analysis & Time-to-Churn Estimation (`src/survival.py`)**:
   - Implements parametric Weibull hazard rate modeling to estimate exact time-to-churn days and survival timeline curves (30, 60, 90, 180 days).

5. **Champion vs. Challenger Shadow Deployments**:
   - Registers retrained models as `challenger` and runs shadow inference alongside `champion`, logging divergence metrics for data-driven promotion (`POST /model/promote`).

6. **Enterprise Monitoring (Prometheus + Slack Webhooks)**:
   - Exposes Prometheus text exposition metrics at `GET /metrics/prometheus`.
   - Dispatches Slack/Webhook alerts (`src/notifications.py`) on data drift threshold breaches ($\ge 0.20$), retraining events, and challenger promotions.

---

## Technical Deep-Dive & Component Functions

### 1. Ingestion Entrypoint ([`api/main.py`](file:///d:/PROJECT%20REPOS/MLOPS/api/main.py))
- **`heal_customer_data()`**: Cleans malformed numerical ranges and performs string similarity matching against valid categorical options.
- **`app.state.model_registry`**: In-memory dictionary holding active domain models, preprocessors, and versions under thread-safe locks (`threading.Lock()`).
- **`POST /domain/bootstrap`**: Dynamically creates isolated directories and initial weights for newly created custom domains.

### 2. Machine Learning Core ([`src/train.py`](file:///d:/PROJECT%20REPOS/MLOPS/src/train.py))
- Runs Optuna hyperparameter optimization over 30 trials with SMOTE oversampling.
- Logs parameters, metrics, artifacts, and registers models under `churn_model_{domain_id}` in MLflow.
