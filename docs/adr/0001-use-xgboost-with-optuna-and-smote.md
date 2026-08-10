# ADR 0001: Selection of XGBoost with Optuna Tuning & SMOTE Class Imbalance Handling

* **Status**: Accepted
* **Date**: 2026-07-30
* **Deciders**: Lead MLOps Engineer & Data Science Team

---

## Context and Problem Statement

Customer churn prediction datasets (e.g. Telco, E-Commerce, School dropout) feature tabular structured data characterized by non-linear interactions, numerical/categorical feature mixtures, and significant class imbalance (churn rate typically ranges between 15% and 25%). 

We needed an inference engine that balances high predictive accuracy ($F1 \ge 0.80$, $\text{ROC-AUC} \ge 0.88$), low latency ($< 50\text{ms}$ per prediction), fast retraining, and local explainability (SHAP value computation).

## Decision Drivers

* **Predictive Performance on Tabular Data**: Tree-based gradient boosting models consistently outperform deep neural networks on tabular datasets without requiring extensive architecture tuning.
* **Explainability Requirements**: High-stakes retention actions (e.g. issuing 10% loyalty discounts or sending academic counselors) require feature-level attribution (TreeSHAP).
* **Imbalance Handling**: Severe class imbalance causes naive classifiers to under-predict high-risk churners.
* **Hyperparameter Optimization Speed**: Retraining loops must run asynchronously without blocking API serving.

## Considered Options

1. **Option 1**: Deep Neural Networks (MLP / TabNet with PyTorch)
2. **Option 2**: Logistic Regression / Random Forest baseline
3. **Option 3**: **XGBoost Classifier + Optuna (30 Trials) + SMOTE Oversampling** (Selected)

## Decision Outcome

**Chosen Option**: Option 3 — **XGBoost Classifier with Optuna Hyperparameter Optimization and SMOTE Oversampling**.

### Rationale

* **XGBoost**: Provides fast execution, built-in missing value handling, and native support for `shap.TreeExplainer` for microsecond-level feature attribution.
* **SMOTE (Synthetic Minority Over-sampling Technique)**: Applied inside an `imblearn.pipeline.Pipeline` during training to balance positive and negative churn samples without leaking evaluation target statistics.
* **Optuna**: Automated hyperparameter tuning optimizing validation $F1$-score over 30 trials per retraining loop.

## Consequences

* **Positive**:
  - Achieves $F1 = 0.83$ and $\text{ROC-AUC} = 0.89$ across industry domain baselines.
  - Sub-50ms inference latency for single-item predictions.
  - Native integration with SHAP feature rankings for domain retention playbooks.
* **Negative**:
  - Requires `joblib` model artifact storage per isolated domain.
  - Memory overhead during Optuna hyperparameter tuning threads.
