# ChurnGuard Changelog & Architecture Progress

All major milestones, architectural refactorings, and feature releases are documented below in chronological order.

---

## Release 2.5.0 (Latest Release)

### Core Features Added
- **Automated Retention Action Playbooks (`src/playbooks.py`)**:
  - Automatically translates SHAP drivers and customer features into domain-tailored intervention playbooks.
  - Supports School, Fitness, Telecom, E-Commerce, and Custom industry domains.
- **Survival Analysis & Time-to-Churn Estimation (`src/survival.py`)**:
  - Implements parametric Weibull hazard rate curves to calculate estimated time-to-churn days and 30/60/90/180-day survival probability timelines.
- **Champion vs. Challenger Shadow Deployments**:
  - Added `ShadowPredictionLog` ORM model in `api/database.py`.
  - Added `GET /model/shadow-status` and `POST /model/promote` for data-driven Challenger model promotion.
- **Enterprise Monitoring & Webhooks**:
  - Added `GET /metrics/prometheus` (`api/metrics_prometheus.py`) exporting real-time Prometheus exposition format metrics.
  - Added Slack Webhook dispatcher (`src/notifications.py`) for drift breaches and retraining alerts.

---

## Release 2.0.0

### Core Architectural Refactor
- **Domain-Isolated Multi-Model Architecture (`src/domain_registry.py`)**:
  - Replaced single static model structure with isolated domain directories (`models/telecom/`, `models/school/`, `models/ecommerce/`, `models/fitness/`, `models/custom_*`).
  - Added per-domain Evidently AI reference baselines (`data/baselines/{domain_id}_baseline.csv`).
  - Added `POST /domain/bootstrap` for dynamic custom domain provisioning.
- **Database Schema Tagging (`api/database.py`)**:
  - Added `domain_id` column to `Prediction`, `SelfHealingLog`, and `DriftReport` SQLite tables with automatic migration.
- **SHAP Factor Deduplication ([`src/evaluate.py`](file:///d:/PROJECT%20REPOS/MLOPS/src/evaluate.py))**:
  - Updated SHAP driver extraction to deduplicate mapped One-Hot Encoded features into unique business feature drivers.

---

## Verification & Test Status

- **Unit Test Coverage**: All **20 out of 20 unit tests passed**.
- **Python Compatibility**: Fully verified on Python 3.12 (`C:\Users\Aayan\AppData\Local\Programs\Python\Python312\python.exe`).
