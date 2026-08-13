"""
FastAPI application entrypoint.

Wires the routers together and owns application startup. The routes themselves
live in api/routers/, and the work they delegate to lives in api/services/.
"""

import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI

from api.config import settings
from api.database import init_db
from api.metrics_prometheus import router as prometheus_router
from api.routers import models as models_router
from api.routers import monitoring as monitoring_router
from api.routers import predictions as predictions_router
from api.routers import self_healing as self_healing_router
from api.services import model_registry

# Re-exported for callers that predate the split into routers and services.
from api.dependencies import verify_api_key, verify_scope  # noqa: F401
from api.security import (  # noqa: F401
    API_KEY,
    HASHED_API_KEY_SCOPES,
    _hash_key,
    generate_high_entropy_key,
)
from api.services.drift_service import run_drift_check  # noqa: F401
from api.services.retrain_service import run_self_healing_retraining  # noqa: F401
from src.domains import get_domain_spec
from src.healing import heal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def heal_customer_data(
    raw_data: dict, domain_id: str = "telecom"
) -> tuple[dict, list[str]]:
    """
    Repair a dirty inbound record against its domain's schema.

    The rules live in src/domains/; this remains as the entry point older
    callers import. Defaults to telecom so their behavior is unchanged.
    """
    return heal(raw_data, get_domain_spec(domain_id))


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_registry.initialize(app)

    app.state.prediction_counter = 0
    app.state.drift_check_every = settings.drift_check_every
    app.state.retraining_status = "idle"
    init_db()

    yield


app = FastAPI(
    title="ChurnGuard API",
    description="Production-grade, self-healing MLOps churn prediction microservice",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(prometheus_router)
app.include_router(monitoring_router.router)
app.include_router(predictions_router.router)
app.include_router(self_healing_router.router)
app.include_router(models_router.router)
