# App Flow Document
## ChurnGuard — Customer Churn Prediction Platform

**Version:** 1.0  
**Last Updated:** June 2026

---

## 1. User Flows

### Flow 1 — Executive Opens Dashboard (Primary Daily Flow)

```
User opens browser → Streamlit Dashboard URL
        │
        ▼
Dashboard loads
        │
        ├── API call: GET /health → confirm API is live
        ├── DB query: fetch last 30 days of predictions
        ├── API call: GET /drift/status → get drift flag
        │
        ▼
Executive Summary Cards render:
  [Total Customers Scored] [% High Risk] [% Medium Risk] [Churn Rate Trend ↑/↓]
        │
        ▼
Drift Alert Banner (if drift detected → yellow/red banner at top)
        │
        ▼
Risk Distribution Chart (donut: High / Medium / Low)
        │
        ▼
Customer Risk Table (sortable, searchable)
  Columns: Customer ID | Risk Score | Risk Tier | Top Risk Factor | Date Scored
        │
        ▼
Executive clicks a customer row
        │
        ▼
Customer Detail Panel opens (sidebar or modal)
  ├── Risk score gauge (0–100)
  ├── SHAP waterfall chart (top 5 factors)
  ├── Plain-English explanation: "This customer is at high risk mainly
  │   because they are on a month-to-month contract (3 months tenure)
  │   with no online security add-on."
  └── Suggested action (Post-MVP): "Offer 20% discount on annual plan"
```

---

### Flow 2 — Upload CSV for Bulk Scoring

```
Executive clicks "Upload & Score" tab
        │
        ▼
File upload widget appears
  "Upload your customer CSV file (max 10MB)"
        │
        ▼
Executive selects file
        │
        ▼
Frontend validates: file is .csv, size < 10MB
        │
  ┌─────┴──────┐
  │ Invalid    │ Valid
  ▼            ▼
Error msg   POST /upload (multipart form)
shown           │
                ▼
            API parses CSV with Pandas
                │
                ▼
            Schema validation
                │
        ┌───────┴───────┐
        │ Missing cols  │ Schema OK
        ▼               ▼
    Error response   Run preprocessing pipeline
    (list of         on all rows
    missing cols)        │
                         ▼
                    Run model.predict_proba()
                    on all rows
                         │
                         ▼
                    Compute SHAP top factor
                    per row (fast: use mean SHAP)
                         │
                         ▼
                    Log all predictions to DB
                         │
                         ▼
                    Return predictions CSV
                    (original columns + appended:
                     churn_probability, risk_tier,
                     top_factor)
                         │
                         ▼
                Dashboard shows:
                ├── Progress bar → "Scoring 1,200 customers..."
                ├── Summary: "847 Low / 241 Medium / 112 High risk"
                └── Download button → predictions.csv
```

---

### Flow 3 — API Integration (CRM / Developer)

```
CRM system or developer script
        │
        ▼
POST /predict
Headers: { "X-API-Key": "your-key" }
Body: { customer JSON object }
        │
        ▼
FastAPI receives request
        │
        ▼
Pydantic validation
        │
  ┌─────┴──────┐
  │ Invalid    │ Valid
  ▼            ▼
422 Error    Load model from MLflow registry
(field       (cached in memory — loaded once at startup)
errors)          │
                 ▼
             Run preprocessing pipeline
             (same pipeline as training)
                 │
                 ▼
             model.predict_proba() → probability
                 │
                 ▼
             SHAP TreeExplainer → top 3 factors
                 │
                 ▼
             Determine risk tier
             (Low / Medium / High by threshold)
                 │
                 ▼
             Log to predictions DB
                 │
                 ▼
             Return JSON response
             { probability, risk_tier,
               top_factors, model_version,
               prediction_id, timestamp }
```

---

### Flow 4 — Model Drift Detection (Background)

```
Every 100 new predictions logged to DB
        │
        ▼
drift.py triggered (can be cron or counter-based)
        │
        ▼
Load last 500 prediction inputs from DB
        │
        ▼
Load training data reference from DVC
        │
        ▼
Evidently DataDriftPreset.run()
(compares incoming data to training distribution)
        │
        ▼
Generate HTML report → save to /reports/drift_{timestamp}.html
Log to drift_reports table in DB
        │
        ▼
Check PSI score for top 5 features
        │
  ┌─────┴──────────────┐
  │ PSI < 0.1          │ PSI 0.1–0.2        PSI > 0.2
  ▼                    ▼                    ▼
No drift            Mild drift           Significant drift
Dashboard:          Dashboard:           Dashboard:
Green banner        Yellow banner        Red banner +
"Data healthy"      "Minor drift"        "Retrain recommended"
                                         Log alert to DB
```

---

## 2. System Startup Flow

```
docker-compose up
        │
        ├── MLflow server starts (port 5000)
        │       └── Loads experiment history from ./mlruns
        │
        ├── FastAPI starts (port 8000)
        │       ├── Load model from MLflow registry ("Production" stage)
        │       ├── Load preprocessing pipeline (joblib)
        │       ├── Initialize SQLite DB (create tables if not exist)
        │       ├── Run /health self-check
        │       └── Ready → log "ChurnGuard API v1.0 running"
        │
        └── Streamlit starts (port 8501)
                ├── Connect to FastAPI (GET /health)
                ├── Load last 30 days predictions from API
                └── Render dashboard
```

---

## 3. Error Handling Flows

### Model Not Loaded
```
API startup → MLflow registry call fails
        │
        ▼
Log error: "Model not found in registry"
Return 503 on all /predict calls
Dashboard shows: "⚠ Model offline — contact admin"
```

### CSV Schema Mismatch
```
POST /upload → Pandas reads CSV
        │
        ▼
Check required columns present
        │
Missing columns detected
        │
        ▼
Return 422: {
  "error": "Missing required columns",
  "missing": ["tenure", "MonthlyCharges"],
  "found": ["customerID", "Contract", ...]
}
Dashboard shows error with column list
```

### Prediction Input Validation Failure
```
POST /predict → Pydantic validation
        │
        ▼
Field "tenure" = -5 (invalid)
        │
        ▼
Return 422: {
  "detail": [
    {
      "loc": ["body", "tenure"],
      "msg": "ensure this value is greater than or equal to 0",
      "type": "value_error.number.not_ge"
    }
  ]
}
```

---

## 4. Navigation Map

```
ChurnGuard Dashboard
├── 🏠 Overview (default)
│   ├── KPI cards
│   ├── Risk distribution chart
│   ├── Churn trend (30-day)
│   └── Drift status banner
│
├── 👥 Customers
│   ├── Searchable risk table
│   └── Customer detail panel (click to open)
│
├── 📁 Upload & Score
│   ├── File upload widget
│   ├── Progress + summary
│   └── Download results
│
└── 📊 Model Health (Admin)
    ├── F1 / AUC metrics
    ├── Precision-recall curve
    ├── Drift report viewer
    └── Prediction volume over time
```
