# ADR 0004: Champion vs. Challenger Shadow Deployment Pattern with Authenticated Promotion Gate

* **Status**: Accepted
* **Date**: 2026-07-30
* **Deciders**: Infrastructure & Reliability Architecture Team

---

## Context and Problem Statement

Automated retraining loops generate updated model pipelines when data drift occurs. Immediately replacing the active live production model ("hot-reloading into Champion slot") introduces significant risk: a newly trained model might regress on key metric boundaries or exhibit unvetted predictions on specific customer subgroups.

We required a zero-risk deployment architecture that allows retrained models to be evaluated against live production traffic without affecting real-time user predictions.

## Decision Drivers

* **Zero-Downtime Reliability**: Production API traffic must always be served by a verified Champion model.
* **Empirical Divergence Tracking**: System must log probability divergence ($\Delta = |P_{\text{champ}} - P_{\text{chall}}|$) on real production requests before model swap.
* **Controlled Promotion Security**: Model promotion must be explicit, authenticated (`X-API-Key`), and audited via Slack webhooks and database logs.

## Considered Options

1. **Option 1**: Direct In-Memory Hot-Reloading (Overwrites Champion model immediately upon retraining completion).
2. **Option 2**: External Blue/Green Container Deployment via Load Balancer.
3. **Option 3**: **In-Process Champion/Challenger Shadow Deployment with Authenticated `POST /model/promote` Gate** (Selected)

## Decision Outcome

**Chosen Option**: Option 3 — **In-Process Champion/Challenger Shadow Deployment Pattern**.

### Architectural Design

1. **Retraining Worker Target**: Retrained model pipelines land under `model_registry[domain_id]["challenger"]`.
2. **Inference Execution**: `run_single_prediction()` scores the payload using the active **Champion** model. If a **Challenger** model is registered, shadow inference scores the item asynchronously and logs $\Delta$ to the `shadow_predictions` database table.
3. **Observation & Status**: Admins observe divergence metrics via `GET /model/shadow-status` or the Streamlit Executive Dashboard.
4. **Gated Promotion**: Invoking `POST /model/promote` with valid API key credentials atomically swaps the Challenger into the Champion position, dispatches a Slack alert, and updates audit logs.

## Consequences

* **Positive**:
  - Eliminates production regression risks from automated retraining loops.
  - Empirically quantifies candidate model behavior on real production requests before promotion.
  - Ensures administrative write actions (`/trigger-retrain`, `/bootstrap`, `/promote`) are secured with API key authorization.
* **Negative**:
  - Requires transient memory allocation for both Champion and Challenger model objects per domain during shadow evaluation windows.
