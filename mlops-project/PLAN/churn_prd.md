# Product Requirements Document (PRD)
## ChurnGuard — Customer Churn Prediction Platform

**Version:** 1.0  
**Author:** Solo Developer  
**Status:** Draft  
**Last Updated:** June 2026

---

## 1. Overview

### 1.1 Product Summary
ChurnGuard is a business-facing web application that predicts which customers are at risk of churning, explains why, and gives business executives actionable insights to retain them — before it's too late.

### 1.2 Problem Statement
Telecom and SaaS companies lose 15–25% of their customer base annually to churn. The average cost to acquire a new customer is 5–7x the cost of retaining an existing one. Currently, businesses react to churn after it happens. ChurnGuard shifts this from reactive to predictive.

### 1.3 Product Vision
A non-technical business executive should be able to open the dashboard, understand which customers are at risk today, see why they're at risk, and take action — all within 5 minutes, with zero data science knowledge required.

### 1.4 Success Metrics
| Metric | Target |
|---|---|
| Model F1 Score (churn class) | ≥ 0.82 |
| Model AUC-ROC | ≥ 0.88 |
| API prediction latency (p95) | < 200ms |
| Dashboard load time | < 3 seconds |
| Data drift alert time | < 1 hour of drift detection |

---

## 2. Users & Personas

### Primary User — Business Executive / Customer Success Manager
- Non-technical; reads dashboards, not code
- Wants to know: "Who is about to leave and why?"
- Makes retention decisions: discounts, outreach campaigns, contract offers
- Uses the app daily or weekly for strategic planning

### Secondary User — ML Engineer (You)
- Monitors model health, drift, retraining triggers
- Accesses the API directly for integration with CRM systems
- Needs clear logging, metrics, and alerting

---

## 3. Core Features

### 3.1 Must Have (MVP)
| ID | Feature | Description |
|---|---|---|
| F01 | Churn Risk Dashboard | Overview of total customers, high/medium/low risk segments, churn rate trend |
| F02 | Individual Customer Risk Score | Score 0–100 with risk tier (High / Medium / Low) |
| F03 | SHAP Explanation | Top 3 factors driving each customer's churn risk in plain English |
| F04 | Bulk CSV Upload | Upload customer data file → get predictions for all rows |
| F05 | REST API (/predict) | Single prediction endpoint for CRM integration |
| F06 | REST API (/predict/batch) | Batch prediction endpoint |
| F07 | Model Performance Page | F1, AUC, precision-recall curve visible to admin |
| F08 | Data Drift Monitoring | Alert banner when input data drifts from training distribution |

### 3.2 Should Have (Post-MVP)
| ID | Feature | Description |
|---|---|---|
| F09 | Customer Segment Filters | Filter dashboard by contract type, tenure group, monthly spend |
| F10 | Retention Action Suggestions | Auto-suggest retention action based on top churn driver |
| F11 | Email Alerts | Send weekly high-risk customer report to executive email |
| F12 | Prediction History | Log all predictions with timestamps for auditing |

### 3.3 Won't Have (Out of Scope v1.0)
- User authentication / multi-tenant support
- Real-time streaming predictions
- CRM direct integration (Salesforce, HubSpot)
- A/B testing of retention strategies

---

## 4. Functional Requirements

### 4.1 Prediction Engine
- Model must accept 20 input features matching Telco dataset schema
- Output must include: churn probability (float 0–1), risk tier (string), top 3 SHAP factors
- Invalid inputs must return HTTP 422 with field-level error messages
- Model version must be logged with every prediction response

### 4.2 Dashboard
- Must display: total customers scored, % high risk, % medium risk, % low risk
- Churn trend chart: 30-day rolling prediction history
- Sortable customer risk table with search by customer ID
- SHAP waterfall chart on customer detail view

### 4.3 File Upload
- Accept CSV files up to 10MB
- Validate schema before processing — return clear error if columns missing
- Return downloadable CSV with predictions appended
- Show progress indicator for files > 500 rows

### 4.4 Monitoring
- Evidently drift report generated on every 100 new predictions
- Dashboard banner: green (no drift), yellow (mild drift), red (significant drift)
- Drift report downloadable as HTML

---

## 5. Non-Functional Requirements

| Category | Requirement |
|---|---|
| Performance | API p95 latency < 200ms for single predictions |
| Reliability | API uptime > 99% (Render free tier acceptable for portfolio) |
| Scalability | Batch endpoint handles up to 5,000 rows per request |
| Security | No PII stored in logs; prediction inputs purged after 30 days |
| Portability | Entire system runs with `docker-compose up` — one command |
| Observability | All predictions logged to SQLite with timestamp, input hash, output |

---

## 6. Constraints
- Solo developer — no team, no budget
- Free-tier cloud deployment (Render, Railway, Streamlit Cloud)
- Dataset: IBM Telco Customer Churn (Kaggle) — 7,043 rows, 21 features
- Build time: 3–4 weeks part-time

---

## 7. Assumptions
- Input data follows the Telco dataset schema (can be adapted for other schemas in v2)
- Business users access the dashboard via desktop browser (mobile not required for v1)
- No real-time data pipeline — predictions run on demand or scheduled batch
