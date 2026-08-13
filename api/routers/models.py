"""
Model lifecycle routes: domain bootstrap, shadow evaluation, promotion.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from api.database import get_db, get_shadow_stats, log_self_healing_event
from api.dependencies import verify_scope
from api.services import model_registry
from src.domain_registry import (
    bootstrap_custom_domain,
    load_domain_model,
    load_domain_preprocessor,
    sanitize_domain_id,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/domain/bootstrap", dependencies=[Depends(verify_scope("admin:bootstrap"))]
)
def bootstrap_domain_endpoint(request: Request, payload: dict):
    domain_name = payload.get("domain_name")
    if not domain_name:
        raise HTTPException(status_code=400, detail="domain_name is required")

    domain_key = bootstrap_custom_domain(domain_name)

    # Bootstrapping writes the domain's baseline, which is what its spec is
    # inferred from — drop any spec cached before that file existed.
    from src.domains import reset_spec_cache

    reset_spec_cache()

    model_registry.register(
        request.app,
        domain_key,
        load_domain_model(domain_key),
        load_domain_preprocessor(domain_key),
        f"{domain_key}-v1",
    )

    return {
        "status": "success",
        "domain_key": domain_key,
        "message": f"Domain '{domain_name}' bootstrapped and isolated.",
    }


@router.get("/model/shadow-status")
def shadow_status(domain: str = "telecom", db: Session = Depends(get_db)):
    domain_key = sanitize_domain_id(domain)
    return get_shadow_stats(db, domain_id=domain_key)


@router.post("/model/promote", dependencies=[Depends(verify_scope("admin:promote"))])
def promote_challenger(
    request: Request, domain: str = "telecom", db: Session = Depends(get_db)
):
    domain_key = sanitize_domain_id(domain)
    promoted_version = model_registry.promote_challenger(request.app, domain_key)
    if promoted_version is None:
        return {
            "status": "info",
            "message": f"No challenger model active for domain '{domain_key}'. Current champion is active.",
        }

    log_self_healing_event(
        db,
        "retraining",
        f"Promoted Challenger model to Champion for domain '{domain_key}'.",
        domain_id=domain_key,
    )

    from src.notifications import send_slack_alert

    send_slack_alert(
        "promotion",
        f"Model Promoted for domain {domain_key}",
        f"Challenger model version {promoted_version} promoted to Champion.",
        domain_id=domain_key,
    )

    return {
        "status": "success",
        "message": f"Challenger promoted to Champion for domain '{domain_key}'.",
    }
