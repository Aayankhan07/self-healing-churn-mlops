# Product Requirements Document (PRD)
## ChurnGuard — Multi-Domain Self-Healing MLOps Churn Platform

**Version:** 2.5  
**Author:** Lead MLOps Engineer  
**Status:** Production  
**Last Updated:** July 2026

---

## 1. Overview

### 1.1 Product Summary
ChurnGuard is an enterprise MLOps churn prediction and retention platform that serves multiple industry domains (Telecom, K-12 Schools, E-Commerce, Fitness Clubs, and Custom domains). It combines rule-based data self-healing, parametric survival analysis, domain-tailored retention playbooks, Champion vs. Challenger A/B testing, and Prometheus monitoring.

### 1.2 Success Metrics
| Metric | Target | Achieved |
|---|---|---|
| Model F1 Score (domain average) | ≥ 0.82 | 0.83 - 0.87 |
| Model AUC-ROC | ≥ 0.88 | 0.89 - 0.94 |
| API prediction latency (p95) | < 200ms | < 45ms |
| Dashboard load time | < 3 seconds | < 1.2s |
| Unit Test Pass Rate | 100% | 20 / 20 Passing |

---

## 2. Core Architectural Capabilities

### 2.1 Multi-Model Domain Isolation
- Maintains isolated model directories for each domain (`models/telecom/`, `models/school/`, `models/ecommerce/`, `models/fitness/`, `models/custom_*`).
- Operates dedicated reference baselines (`data/baselines/{domain_id}_baseline.csv`).
- Provisions custom domains dynamically via `POST /domain/bootstrap`.

### 2.2 Ingestion & Retraining Data Self-Healing
- Implements rule-based correction (clamping negative tenures, imputing missing charges, norming categorical variables) and dynamic typo resolution using string similarity metrics (`difflib.get_close_matches`).
- Logs database correction events prior to retraining.

### 2.3 Automated Retention Action Playbooks (`src/playbooks.py`)
- Maps active domain + SHAP top drivers + customer features to actionable intervention recommendations.
- **School**: *Schedule mandatory Academic Counselor check-in and notify parents via portal.*
- **Fitness**: *Send automated SMS offering a complimentary 1-on-1 Personal Trainer session.*
- **Telecom**: *Trigger proactive 10% loyalty discount on 1-year contract pitch.*

### 2.4 Survival Analysis & Time-to-Churn Estimation (`src/survival.py`)
- Calculates parametric Weibull hazard rate curves, estimating exact time-to-churn days and survival timeline probabilities (30, 60, 90, 180 days).

### 2.5 Champion vs. Challenger Shadow Deployments
- Logs shadow inference metrics in `shadow_predictions` table.
- Exposes `GET /model/shadow-status` and `POST /model/promote` for data-driven Challenger promotion.

### 2.6 Enterprise Monitoring (Prometheus + Slack Webhooks)
- Exposes Prometheus metrics at `GET /metrics/prometheus`.
- Dispatches Slack/Webhook notifications (`src/notifications.py`) for data drift breaches ($\ge 0.20$), retraining events, and model promotions.
