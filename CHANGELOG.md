# ChurnGuard Release & Platform Changelog

All notable changes, architectural decisions, security enhancements, and MLOps pipeline upgrades are documented here.

---

## [v1.3.0] - 2026-07-30

### 🎨 Dashboard UI/UX & Frontend Polish (`dashboard/app.py`)
- **Clean Professional Design System (No Emojis)**:
  - Redesigned executive dashboard using a dark-mode glassmorphism palette (`#030712` obsidian background, `#0F172A` card surface, cyan-teal accents).
  - Replaced all emojis across navigation tabs, headers, badges, and control buttons with clean text labels (`Telecom`, `K-12 School`, `E-Commerce`, `Fitness`).
- **1-Click Profile Presets**:
  - Added interactive `Load High-Risk Customer Preset` and `Load Loyal Customer Preset` buttons on the Customers page to auto-fill input attributes for fast testing.
- **EEOC Four-Fifths Subgroup Fairness Display**:
  - Upgraded **Model Health** tab to render a 4-column metric layout showcasing:
    - EEOC 4/5 Selection Ratio ($\ge 0.800$)
    - Demographic Parity Difference ($\le 0.150$)
    - Equalized Odds Difference ($\le 0.150$)
    - Bias Audit Status (`ACCEPTABLE` / `DISPARITY FLAGGED`)
- **Evidently AI Diagnostics Dark Mode HTML Auto-Upgrader (`api/main.py` & `src/monitor.py`)**:
  - Automatically converts and renders dark-mode high-contrast HTML report cards (`#080C14` background, `#14B8A6` headings, `#10B981` status badges) for data drift diagnostics.
- **Domain Model Loading & Prediction Variable Fixes**:
  - Resolved offline fallback model resolution by routing to domain-isolated model paths via `load_domain_model(current_domain_id)`.
  - Fixed `NameError` by unifying prediction response object references (`res.get(...)`).

---

## [v1.2.0] - 2026-07-30

### 🚀 Added
- **Subgroup Fairness & Demographic Bias Analysis Engine (`src/evaluate.py`)**:
  - Computes Selection Rate, Recall (TPR), False Positive Rate (FPR), and F1-Score across sensitive demographic subgroups (`SeniorCitizen`, `Contract`, `TenureBucket`).
  - Dual-gated fairness evaluation enforcing both scale-dependent **EEOC Four-Fifths Selection Ratio** ($\frac{\text{Rate}_{\min}}{\text{Rate}_{\max}} \ge 0.80$) and flat **Demographic Parity Difference** ($\le 0.150$).
- **Role-Based Access Control (RBAC) (`api/main.py`)**:
  - Implemented granular scope authorization (`read:predict`, `write:retrain`, `admin:bootstrap`, `admin:promote`).
  - Added cryptographic 256-bit high-entropy token generator (`secrets.token_urlsafe(32)`).
  - Encrypted and verified API keys using **SHA-256 key hashing** (`_hash_key()`) to prevent plaintext exposure.
- **Runtime Promotion & EEOC Fairness Gate**:
  - Automated evaluation script (`scripts/evaluate_challenger_gate.py`) and mirrored GitHub Actions workflow (`.github/workflows/challenger_eval_gate.yml`).
  - Blocks model promotion if candidate Challenger regresses $>1\%$ in $F1$/$\text{ROC-AUC}$ or breaches EEOC Four-Fifths fairness bounds.
- **ONNX Model Synchronization on Promotion (`api/main.py`)**:
  - Synchronizes active ONNX Runtime inference sessions upon Challenger promotion via `POST /model/promote`, eliminating stale model prediction leaks.
- **Latency Benchmarking & Load Testing Suite**:
  - Created Locust load testing suite (`scripts/locustfile.py`) and automated latency benchmark tool (`scripts/benchmark_latency.py`).
- **Infrastructure as Code (Terraform) (`terraform/`)**:
  - AWS ECS Fargate, RDS PostgreSQL, S3 Model Bucket, ALB, S3 Remote State Backend (`churnguard-tf-state`), DynamoDB State Lock (`churnguard-tf-locks`), and **AWS Secrets Manager**.
- **Prometheus & Grafana Observability**:
  - Auto-provisioned Grafana observability dashboard (`grafana/dashboards/churnguard_observability.json`).

---

## [v1.1.0] - 2026-07-15
- Initial Champion/Challenger shadow deployment pattern and SQLite operational store launch.
