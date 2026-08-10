"""
Prometheus Exporter Module.
Provides GET /metrics/prometheus endpoint returning Prometheus text exposition metrics.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from api.database import get_db, count_predictions, get_latest_drift
import time

router = APIRouter()
START_TIME = time.time()


@router.get("/metrics/prometheus")
def prometheus_metrics(db: Session = Depends(get_db)):
    """
    Export real-time Prometheus metrics for scrapers.
    """
    total_preds = count_predictions(db)
    telecom_preds = count_predictions(db, domain_id="telecom")
    school_preds = count_predictions(db, domain_id="school")
    ecommerce_preds = count_predictions(db, domain_id="ecommerce")
    fitness_preds = count_predictions(db, domain_id="fitness")

    drift_rec = get_latest_drift(db)
    drift_score = drift_rec.drift_score if drift_rec else 0.0
    uptime = time.time() - START_TIME

    from api.database import get_shadow_stats, risk_distribution, get_self_healing_logs

    shadow_stats = get_shadow_stats(db)
    avg_delta = shadow_stats.get("avg_delta", 0.0)

    risk_dist = risk_distribution(db)
    high_risk = risk_dist.get("High", 0)
    med_risk = risk_dist.get("Medium", 0)
    low_risk = risk_dist.get("Low", 0)

    sh_logs = get_self_healing_logs(db, limit=500)
    sh_count = len(sh_logs)

    lines = [
        "# HELP churnguard_uptime_seconds Total uptime of the API service in seconds",
        "# TYPE churnguard_uptime_seconds counter",
        f"churnguard_uptime_seconds {uptime:.1f}",
        "# HELP churnguard_predictions_total Total predictions processed by ChurnGuard",
        "# TYPE churnguard_predictions_total counter",
        f"churnguard_predictions_total {total_preds}",
        f'churnguard_predictions_domain_total{{domain="telecom"}} {telecom_preds}',
        f'churnguard_predictions_domain_total{{domain="school"}} {school_preds}',
        f'churnguard_predictions_domain_total{{domain="ecommerce"}} {ecommerce_preds}',
        f'churnguard_predictions_domain_total{{domain="fitness"}} {fitness_preds}',
        "# HELP churnguard_drift_score Most recent dataset drift score",
        "# TYPE churnguard_drift_score gauge",
        f"churnguard_drift_score {drift_score:.4f}",
        "# HELP churnguard_shadow_divergence_delta Mean prediction divergence between Champion and Challenger",
        "# TYPE churnguard_shadow_divergence_delta gauge",
        f"churnguard_shadow_divergence_delta {avg_delta:.4f}",
        "# HELP churnguard_risk_tier_count Total predictions by risk tier",
        "# TYPE churnguard_risk_tier_count counter",
        f'churnguard_risk_tier_count{{tier="High"}} {high_risk}',
        f'churnguard_risk_tier_count{{tier="Medium"}} {med_risk}',
        f'churnguard_risk_tier_count{{tier="Low"}} {low_risk}',
        "# HELP churnguard_self_healing_events_total Total self-healing events logged",
        "# TYPE churnguard_self_healing_events_total counter",
        f"churnguard_self_healing_events_total {sh_count}",
    ]

    content = "\n".join(lines) + "\n"
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(content, media_type="text/plain; version=0.0.4")
