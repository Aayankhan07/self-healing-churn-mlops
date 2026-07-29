# ChurnGuard: Multi-Domain Self-Healing MLOps Platform

ChurnGuard is an enterprise MLOps platform designed to predict customer and student churn across multiple industries (Telecom, K-12 Schools, E-Commerce, Fitness, and Custom domains). It features domain-isolated model registries, rule-based data self-healing, automated retention playbooks, parametric survival analysis, Champion vs. Challenger shadow deployments, and Prometheus monitoring.

---

## Key Platform Capabilities

1. **Domain-Isolated Multi-Model Registry**:
   * Operates distinct XGBoost pipelines and preprocessors for each industry domain (`models/telecom/`, `models/school/`, `models/ecommerce/`, `models/fitness/`, `models/custom_*`).
   * Evaluates Evidently AI data drift against isolated per-domain baselines (`data/baselines/{domain_id}_baseline.csv`).
   * Provides dynamic custom domain provisioning via `POST /domain/bootstrap`.

2. **Automated Retention Action Playbooks (`src/playbooks.py`)**:
   * Translates top SHAP drivers into domain-tailored intervention playbooks.
   * **School Student Churn**: *Schedule mandatory Academic Counselor check-in and notify parents via portal.*
   * **Fitness Member Churn**: *Send automated SMS offering a complimentary 1-on-1 Personal Trainer session.*
   * **Telecom Customer Churn**: *Trigger proactive 10% loyalty discount on 1-year contract pitch.*

3. **Survival Analysis & Time-to-Churn Estimation (`src/survival.py`)**:
   * Implements parametric Weibull hazard rate modeling to estimate exact time-to-churn days and survival timelines (30, 60, 90, 180 days).

4. **Champion vs. Challenger Shadow Deployments**:
   * Retrained models are registered as `challenger` and run in shadow inference alongside `champion`.
   * Tracks divergence metrics via `GET /model/shadow-status` and supports data-driven promotion via `POST /model/promote`.

5. **Enterprise Monitoring & Webhook Alerts**:
   * Exposes Prometheus text-formatted metrics at `GET /metrics/prometheus`.
   * Dispatches instant Slack/Webhook alerts (`src/notifications.py`) for drift breaches ($\ge 0.20$), database self-healing events, and challenger promotions.

---

## Repository Structure

```text
├── api/                     # FastAPI backend application
│   ├── database.py          # SQLAlchemy models with domain_id schema tagging
│   ├── main.py              # Application entrypoint & domain routing
│   ├── metrics_prometheus.py# Prometheus exposition metrics endpoint
│   ├── predict.py           # Single & batch prediction runners with playbooks
│   └── schemas.py           # Pydantic data schemas
├── dashboard/               # Streamlit application
│   └── app.py               # Dark-mode executive dashboard & domain switcher
├── data/                    # Storage for raw datasets and domain baselines
├── dvc.yaml                 # DVC pipeline stages
├── metrics/                 # Outputs of evaluation stages (JSON)
├── mlflow.db                # SQLite database for MLflow runs
├── models/                  # Domain-isolated model directories
│   ├── ecommerce/
│   ├── fitness/
│   ├── school/
│   └── telecom/
├── params.yaml              # Configuration thresholds
├── scripts/                 # Utility scripts (e.g. train_all_domains.py)
├── src/                     # Core Python modules
│   ├── domain_registry.py   # Domain artifact & baseline manager
│   ├── evaluate.py          # SHAP explanations & deduplicated driver extraction
│   ├── features.py          # Pipeline preprocessor builder
│   ├── monitor.py           # Evidently AI drift detector
│   ├── notifications.py     # Slack webhook notification dispatcher
│   ├── playbooks.py         # Domain retention playbook engine
│   ├── survival.py          # Parametric survival curve calculator
│   └── train.py             # Optuna tuning & MLflow registration pipeline
└── tests/                   # Pytest testing suite (20/20 passing)
```

---

## Quickstart Guide

### 1. Setup Virtual Environment & Install Dependencies
Ensure you are using **Python 3.10+** (tested and optimized for Python 3.12):

```powershell
C:\Users\Aayan\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run Automated Testing Suite
```powershell
C:\Users\Aayan\AppData\Local\Programs\Python\Python312\python.exe -m pytest -v
```

### 3. Launch Services

#### Start FastAPI Backend Engine:
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

#### Start Streamlit Dashboard:
```powershell
streamlit run dashboard/app.py --server.port 8501
```

- **Dashboard UI**: `http://localhost:8501`
- **FastAPI Swagger Docs**: `http://localhost:8000/docs`
- **Prometheus Metrics**: `http://localhost:8000/metrics/prometheus`
