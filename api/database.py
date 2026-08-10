"""
SQLAlchemy + SQLite setup. Lightweight — no external DB needed.
"""

from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timedelta
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./churnguard.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(String, primary_key=True)
    customer_id = Column(String, nullable=True)
    domain_id = Column(String, default="telecom", index=True)
    input_hash = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)
    model_ver = Column(String, nullable=False)
    features_json = Column(String, nullable=True)
    healed_actions = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class SelfHealingLog(Base):
    __tablename__ = "self_healing_logs"
    id = Column(String, primary_key=True)
    domain_id = Column(String, default="telecom", index=True)
    event_type = Column(String, nullable=False)  # "data_quality" or "retraining"
    description = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    __tablename__ = "drift_reports"
    id = Column(String, primary_key=True)
    domain_id = Column(String, default="telecom", index=True)
    report_path = Column(String, nullable=False)
    drift_detected = Column(Integer, nullable=False)
    drift_score = Column(Float)
    n_samples = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


class ShadowPredictionLog(Base):
    __tablename__ = "shadow_predictions"
    id = Column(String, primary_key=True)
    domain_id = Column(String, default="telecom", index=True)
    customer_id = Column(String, nullable=True)
    champion_proba = Column(Float, nullable=False)
    challenger_proba = Column(Float, nullable=False)
    probability_delta = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
    from sqlalchemy import text

    with engine.connect() as conn:
        for tbl in ["predictions", "self_healing_logs", "drift_reports"]:
            try:
                conn.execute(
                    text(
                        f"ALTER TABLE {tbl} ADD COLUMN domain_id VARCHAR DEFAULT 'telecom'"
                    )
                )
                conn.commit()
            except Exception:
                pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Query helpers ──────────────────────────────────────────


def log_prediction(
    db,
    prediction_id,
    customer_id,
    input_hash,
    probability,
    risk_tier,
    prediction,
    model_ver,
    features_json=None,
    healed_actions=None,
    domain_id="telecom",
):
    record = Prediction(
        id=prediction_id,
        customer_id=customer_id,
        domain_id=domain_id,
        input_hash=input_hash,
        probability=probability,
        risk_tier=risk_tier,
        prediction=prediction,
        model_ver=model_ver,
        features_json=features_json,
        healed_actions=healed_actions,
    )
    db.add(record)
    db.commit()


def count_predictions(db, domain_id: str = None) -> int:
    query = db.query(Prediction)
    if domain_id:
        query = query.filter(Prediction.domain_id == domain_id)
    return query.count()


def predictions_last_n_days(db, n: int = 30, domain_id: str = None):
    cutoff = datetime.utcnow() - timedelta(days=n)
    query = db.query(Prediction).filter(Prediction.created_at >= cutoff)
    if domain_id:
        query = query.filter(Prediction.domain_id == domain_id)
    return query.all()


def risk_distribution(db, domain_id: str = None) -> dict:
    query = db.query(Prediction)
    if domain_id:
        query = query.filter(Prediction.domain_id == domain_id)
    rows = query.all()
    dist = {"Low": 0, "Medium": 0, "High": 0}
    for r in rows:
        dist[r.risk_tier] = dist.get(r.risk_tier, 0) + 1
    return dist


def last_n_inputs(db, n: int = 500, domain_id: str = None):
    """Return last N prediction records for drift monitoring."""
    query = db.query(Prediction)
    if domain_id:
        query = query.filter(Prediction.domain_id == domain_id)
    return query.order_by(Prediction.created_at.desc()).limit(n).all()


def log_drift_report(
    db,
    report_id,
    report_path,
    drift_detected,
    drift_score,
    n_samples,
    domain_id: str = "telecom",
):
    record = DriftReport(
        id=report_id,
        domain_id=domain_id,
        report_path=report_path,
        drift_detected=int(drift_detected),
        drift_score=drift_score,
        n_samples=n_samples,
    )
    db.add(record)
    db.commit()


def get_latest_drift(db, domain_id: str = None):
    query = db.query(DriftReport)
    if domain_id:
        query = query.filter(DriftReport.domain_id == domain_id)
    return query.order_by(DriftReport.created_at.desc()).first()


def log_self_healing_event(
    db, event_type: str, description: str, domain_id: str = "telecom"
):
    import uuid

    record = SelfHealingLog(
        id=str(uuid.uuid4()),
        domain_id=domain_id,
        event_type=event_type,
        description=description,
        created_at=datetime.utcnow(),
    )
    db.add(record)
    db.commit()


def get_self_healing_logs(db, limit: int = 100, domain_id: str = None):
    query = db.query(SelfHealingLog)
    if domain_id:
        query = query.filter(SelfHealingLog.domain_id == domain_id)
    return query.order_by(SelfHealingLog.created_at.desc()).limit(limit).all()


def log_shadow_prediction(
    db, domain_id: str, customer_id: str, champion_proba: float, challenger_proba: float
):
    import uuid

    delta = abs(champion_proba - challenger_proba)
    record = ShadowPredictionLog(
        id=str(uuid.uuid4()),
        domain_id=domain_id,
        customer_id=customer_id,
        champion_proba=champion_proba,
        challenger_proba=challenger_proba,
        probability_delta=delta,
    )
    db.add(record)
    db.commit()


def get_shadow_stats(db, domain_id: str = "telecom", limit: int = 100):
    rows = (
        db.query(ShadowPredictionLog)
        .filter(ShadowPredictionLog.domain_id == domain_id)
        .order_by(ShadowPredictionLog.created_at.desc())
        .limit(limit)
        .all()
    )
    if not rows:
        return {"sample_count": 0, "avg_delta": 0.0, "status": "no_shadow_logs"}
    avg_delta = sum(r.probability_delta for r in rows) / len(rows)
    return {
        "sample_count": len(rows),
        "avg_delta": round(avg_delta, 4),
        "status": "active_shadow_eval",
    }
