"""
FastAPI application entrypoint.

Wires the routers together and owns application startup. The routes themselves
live in api/routers/, and the work they delegate to lives in api/services/.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import logging  # noqa: E402
import os  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

import mlflow  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from api.database import init_db  # noqa: E402
from api.metrics_prometheus import router as prometheus_router  # noqa: E402
from api.routers import models as models_router  # noqa: E402
from api.routers import monitoring as monitoring_router  # noqa: E402
from api.routers import predictions as predictions_router  # noqa: E402
from api.routers import self_healing as self_healing_router  # noqa: E402
from api.services import model_registry  # noqa: E402

# Re-exported for callers that predate the split into routers and services.
from api.dependencies import verify_api_key, verify_scope  # noqa: E402,F401
from api.security import (  # noqa: E402,F401
    API_KEY,
    HASHED_API_KEY_SCOPES,
    _hash_key,
    generate_high_entropy_key,
)
from api.services.drift_service import run_drift_check  # noqa: E402,F401
from api.services.retrain_service import (  # noqa: E402,F401
    run_self_healing_retraining,
)
from src.domains import get_domain_spec  # noqa: E402
from src.healing import heal  # noqa: E402

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
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

    model_registry.initialize(app)

    app.state.prediction_counter = 0
    app.state.drift_check_every = int(os.getenv("DRIFT_CHECK_EVERY_N", 100))
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
