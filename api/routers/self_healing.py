"""
Self-healing console routes: event log and manual retrain trigger.
"""

import logging
import threading

from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from api.database import get_db, get_self_healing_logs, log_self_healing_event
from api.dependencies import verify_scope
from api.services import model_registry
from api.services.retrain_service import run_self_healing_retraining
from src.domain_registry import sanitize_domain_id

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/self-healing/logs")
def get_sh_logs(
    domain: str = "telecom", limit: int = 100, db: Session = Depends(get_db)
):
    domain_key = sanitize_domain_id(domain)
    logs = get_self_healing_logs(db, limit=limit, domain_id=domain_key)
    return [
        {
            "id": log.id,
            "event_type": log.event_type,
            "description": log.description,
            "created_at": log.created_at.isoformat(),
        }
        for log in logs
    ]


@router.post(
    "/self-healing/trigger-retrain",
    dependencies=[Depends(verify_scope("write:retrain"))],
)
def trigger_retrain(
    request: Request, domain: str = "telecom", db: Session = Depends(get_db)
):
    app = request.app
    domain_key = sanitize_domain_id(domain)
    if not model_registry.claim_retraining_slot(app):
        return {
            "status": "already_running",
            "message": f"Retraining is already running for domain '{domain_key}'.",
        }

    log_self_healing_event(
        db,
        "retraining",
        f"Manually triggered retraining for domain '{domain_key}' from self-healing console.",
        domain_id=domain_key,
    )
    threading.Thread(target=run_self_healing_retraining, args=(app, domain_key)).start()
    return {
        "status": "started",
        "message": f"Asynchronous retraining triggered for domain '{domain_key}'.",
    }
