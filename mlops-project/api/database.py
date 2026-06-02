"""
SQLAlchemy + SQLite setup. Lightweight — no external DB needed.
"""
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, text
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
    input_hash = Column(String, nullable=False)
    probability = Column(Float, nullable=False)
    risk_tier = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)
    model_ver = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class DriftReport(Base):
    __tablename__ = "drift_reports"
    id = Column(String, primary_key=True)
    report_path = Column(String, nullable=False)
    drift_detected = Column(Integer, nullable=False)
    drift_score = Column(Float)
    n_samples = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Query helpers ──────────────────────────────────────────

def log_prediction(db, prediction_id, customer_id, input_hash,
                   probability, risk_tier, prediction, model_ver):
    record = Prediction(
        id=prediction_id, customer_id=customer_id, input_hash=input_hash,
        probability=probability, risk_tier=risk_tier, prediction=prediction,
        model_ver=model_ver
    )
    db.add(record)
    db.commit()


def count_predictions(db) -> int:
    return db.query(Prediction).count()


def predictions_last_n_days(db, n: int = 30):
    cutoff = datetime.utcnow() - timedelta(days=n)
    return db.query(Prediction).filter(Prediction.created_at >= cutoff).all()


def risk_distribution(db) -> dict:
    rows = db.query(Prediction).all()
    dist = {"Low": 0, "Medium": 0, "High": 0}
    for r in rows:
        dist[r.risk_tier] = dist.get(r.risk_tier, 0) + 1
    return dist


def last_n_inputs(db, n: int = 500):
    """Return last N prediction records for drift monitoring."""
    return (db.query(Prediction)
              .order_by(Prediction.created_at.desc())
              .limit(n)
              .all())


def log_drift_report(db, report_id, report_path, drift_detected, drift_score, n_samples):
    record = DriftReport(
        id=report_id, report_path=report_path,
        drift_detected=int(drift_detected),
        drift_score=drift_score, n_samples=n_samples
    )
    db.add(record)
    db.commit()


def get_latest_drift(db):
    return (db.query(DriftReport)
              .order_by(DriftReport.created_at.desc())
              .first())
