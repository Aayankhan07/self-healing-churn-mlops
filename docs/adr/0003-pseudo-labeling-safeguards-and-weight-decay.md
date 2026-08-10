# ADR 0003: Pseudo-Labeling Safeguards, Ratio Capping & Weight Decay for Retraining

* **Status**: Accepted
* **Date**: 2026-07-30
* **Deciders**: MLOps Safety & Ethics Working Group

---

## Context and Problem Statement

When data drift is detected ($\text{drift\_score} \ge 0.20$), the self-healing retraining engine ingests recent production prediction inputs to retrain the domain model. Ground-truth labels (actual customer churn events) often take 30 to 90 days to materialize. 

Naive semi-supervised pseudo-labeling (assigning $\text{Label}=1$ if $P \ge 0.85$ and $\text{Label}=0$ if $P \le 0.15$) risks creating a self-reinforcing bias loop where the model trains on its own past predictions, amplifying existing errors and model overconfidence.

## Decision Drivers

* **Model Calibration Preservation**: The retrained model must not amplify past biases or over-emphasize extreme predictions.
* **Ground-Truth Precedence**: Actual observed customer outcomes must take precedence over high-confidence predictions.
* **Feedback Loop Elimination**: Unlabeled production inputs must not overwhelm the retraining sample pool.

## Considered Options

1. **Option 1**: Retain only verified ground-truth records (Aborts retraining if ground-truth labels are unavailable).
2. **Option 2**: Unconstrained pseudo-labeling on all predictions with $P \ge 0.85$ or $P \le 0.15$.
3. **Option 3**: **Ground-Truth Precedence + 30% Pseudo-Label Ratio Cap + 0.25 Sample Weight Decay** (Selected)

## Decision Outcome

**Chosen Option**: Option 3 — **Multi-Layer Pseudo-Labeling Safeguards**.

### Implementation Rules

1. **Ground-Truth Precedence**: Matches production customer IDs against verified ground-truth tables first.
2. **Ratio Cap (`MAX_PSEUDO_LABEL_RATIO = 0.30`)**: Pseudo-labeled samples are capped at a maximum of 30% of the retraining sample volume.
3. **Weight Decay (`weight = 0.25`)**: High-confidence pseudo-labeled records are assigned a reduced sample weight of `0.25` and a 50% random subsampling rate, preventing overconfident gradient updates during XGBoost fitting.

## Consequences

* **Positive**:
  - Eliminates self-reinforcing bias loops while enabling closed-loop continuous adaptation under data drift.
  - Maintains statistical calibration and fair evaluation metrics across demographic subgroups.
* **Negative**:
  - Requires tracking `true_label_count` vs `pseudo_label_count` in retraining log metadata.
