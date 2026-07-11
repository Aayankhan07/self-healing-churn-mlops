# System Changes & Self-Healing Pipeline Integration

We have completed the design, implementation, and verification of the **Non-Agentic Self-Healing Pipeline (ChurnGuard)** for Python 3.12 on Windows.

Here is a summary of the changes introduced to the codebase:

---

## 1. Database-Level Self-Healing for Retraining
* **Data Cleaning Prior to Retraining**: Updated `run_self_healing_retraining` inside [api/main.py](file:///d:/PROJECT%20REPOS/MLOPS/mlops-project/api/main.py) to dynamically run `heal_customer_data` on each production record retrieved from SQLite. This ensures that the retraining dataset (merged with reference data) is completely clean of typos, negative numeric values, constraint mismatches, and raw SeniorCitizen formats.
* **Audit Logs**: Database-level corrections during retraining are automatically logged to the SQLite event log as `data_quality` events.

## 2. Ingestion-Level Self-Healing (API Gateway)
* **Data Healing Rules**: Implemented `heal_customer_data` in [api/main.py](file:///d:/PROJECT%20REPOS/MLOPS/mlops-project/api/main.py) to correct:
  * *Numeric*: Clamps negative tenure to `0`, imputes missing Monthly/Total charges with median, and recomputes `TotalCharges = MonthlyCharges * tenure` when constraint mismatches occur.
  * *Categorical*: Standardizes `SeniorCitizen` format variants to binary values. Uses `difflib.get_close_matches` to resolve minor typos (e.g. `Electrnic check` mapping to `Electronic check`).
* **Endpoint Upgrades**: Modified the `/predict` and `/predict/batch` endpoints to run ingestion self-healing transparently and log actions in the database.
* **Thread-Safe Hot-Reloading**: Wrapped `(model, preprocessor)` state updates inside a `threading.Lock()` to prevent race conditions during drift-triggered retraining swaps.

## 3. Test Coverage & Validation
* Updated existing API tests in [tests/test_api.py](file:///d:/PROJECT%20REPOS/MLOPS/mlops-project/tests/test_api.py) to expect healed input successes (`200 OK`) instead of validation errors (`422 Unprocessable Entity`).
* Created a comprehensive test suite in [tests/test_self_healing.py](file:///d:/PROJECT%20REPOS/MLOPS/mlops-project/tests/test_self_healing.py) including:
  * Unit tests for numeric clamping, constraint recomputation, dynamic categorical typos, and normalization.
  * Integration tests for `/predict` healing responses.
  * Integration tests verifying database-level self-healing during model retraining execution.

## 4. Python 3.12 Compatibility Fixes
* **Requirements Update**: Relaxed dependency version locks in [requirements.txt](file:///d:/PROJECT%20REPOS/MLOPS/mlops-project/requirements.txt) (e.g., `pydantic>=2.7.1`) to resolve binary compatibility errors.
* **MLflow Package Setup**: Downgraded `setuptools` to version `<81` to keep `pkg_resources` accessible (as required by MLflow v2.13.0).
* **NumPy Compatibility**: Reinstalled `numpy==1.26.4` and downgraded `scipy<1.14` to resolve compatibility issues where SciPy was trying to import the removed `numpy.long` attribute.

---

## 5. Correctness, Security & UI/UX Polish
* **Crash-Safe Retraining Path**: Moved retraining dataset output from the Git-tracked `train.csv` to a temporary `train_retrain.csv` path, using the `TRAIN_DATA_PATH` environment variable in [src/train.py](file:///d:/PROJECT%20REPOS/MLOPS/mlops-project/src/train.py) to prevent baseline training set corruption if retraining is interrupted.
* **Atomic Counter & Thread-Safe Locking**: Incremented prediction counters inside a `threading.Lock()` block and spawned drift checks inside a separate thread to prevent request thread blocks and SQLite transaction conflicts.
* **Active Baseline Reload**: The cached reference dataset (`app.state.reference_data`) now refreshes automatically with the newly combined dataset during hot-reloads.
* **Strict API Key Validation**: FastAPI and Streamlit both validate `API_KEY` on startup when `ENV=production`. If missing, the app fails to start immediately with a `ValueError`.
* **Confirmation Bias Prevention Loop**: Auto-retraining is capped at a maximum of `3` consecutive pseudo-labeled cycles. If no new ground-truth data is resolved across three retraining intervals, retraining is skipped to prevent confirmation bias drift.
* **Precise Fuzzy-Match Cutoff**: Increased `difflib.get_close_matches` cutoff from `0.3` to `0.6` to avoid false-positive categorical value mapping.
* **Model Health Metrics & Drift Synced**:
  * Added `/metrics` API endpoint that queries the live Production model run from MLflow (falling back to local evaluation json files or defaults).
  * Batch prediction endpoint `/predict/batch` now increments the prediction counter and triggers the Evidently AI drift check thread.
  * Streamlit's Model Health tab queries the `/metrics` endpoint dynamically, displaying F1 score, AUC-ROC, and active source.
* **SHAP Driver Value Formatting**: Implemented prefix matching in `get_top_shap_factors` inside [src/evaluate.py](file:///d:/PROJECT%20REPOS/MLOPS/mlops-project/src/evaluate.py) to map One-Hot Encoded feature columns back to original feature names and customer raw values, resolving `"N/A"` displays.
* **Logging Duplication Prevention**: The retraining loop checks if a log description containing the customer's self-healing action already exists in SQLite before writing new logs, avoiding duplicate telemetry clutter.

---

## 6. Verification Command
You can run the entire test suite locally under Python 3.12 with:
```powershell
pytest tests/
```

All tests pass cleanly:
```
tests\test_api.py .......                                                [ 53%]
tests\test_self_healing.py ......                                        [100%]
======================= 13 passed, 24 warnings in 1.37s =======================
```
