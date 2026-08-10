# ADR 0002: Embedded SQLite Operational Store with SQLAlchemy ORM Abstraction

* **Status**: Accepted
* **Date**: 2026-07-30
* **Deciders**: Platform Architecture Team

---

## Context and Problem Statement

ChurnGuard requires a persistent storage layer to record real-time prediction logs, data-healing audit trails, data drift reports, and shadow deployment metrics ($\Delta = |P_{\text{champ}} - P_{\text{chall}}|$). 

We required a lightweight database setup that supports zero-dependency local development and automated CI testing while preserving an enterprise path to cloud relational databases (e.g. AWS RDS PostgreSQL or Cloud SQL).

## Decision Drivers

* **Zero-Dependency Local DX**: Developers and automated pytest runners must be able to clone the repository and run all API & dashboard services without setting up Docker containers or external DB instances.
* **ORM Abstraction**: Database queries must use standard SQLAlchemy ORM models so that environment variable configurations (`DATABASE_URL`) can switch the engine to PostgreSQL or MySQL without altering business logic.
* **Concurrency Requirements**: Write volume is read-heavy with periodic batch prediction writes and retraining log inserts.

## Considered Options

1. **Option 1**: Local JSON / Flat CSV File Persistence
2. **Option 2**: Mandatory External PostgreSQL Instance
3. **Option 3**: **Embedded SQLite (`churnguard.db`) with SQLAlchemy ORM (`check_same_thread=False`)** (Selected)

## Decision Outcome

**Chosen Option**: Option 3 — **Embedded SQLite with SQLAlchemy ORM**.

### Rationale

* SQLite operates as a single file (`churnguard.db`), enabling instant startup, seamless testing with pytest, and deterministic state resets.
* Using SQLAlchemy ORM (`Prediction`, `SelfHealingLog`, `DriftReport`, `ShadowPredictionLog`) decouples table schemas from raw SQL syntax. Deployments in cloud environments (ECS / Cloud Run) simply override `DATABASE_URL=postgresql://user:pass@rds.amazonaws.com/churnguard`.

## Consequences

* **Positive**:
  - Zero external database dependencies for local execution and Pytest test runs.
  - Complete portability across Windows, macOS, and Linux.
  - Direct PostgreSQL compatibility via SQLAlchemy ORM.
* **Negative**:
  - SQLite write concurrency is limited by file locking under multi-worker Uvicorn setups (handled via single-writer thread locks or setting `DATABASE_URL` to PostgreSQL in high-throughput production environments).
