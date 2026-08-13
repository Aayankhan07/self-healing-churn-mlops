"""
Access to the in-memory domain model registry.

Every read and write of `app.state.model_registry` goes through here, always
under `app.state.model_lock`. Before this module three different call sites
reached into the registry with three different locking disciplines, which is
how a challenger could be swapped out from under a request mid-prediction.
"""

import logging
import threading
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DOMAINS = ["telecom", "school", "ecommerce", "fitness"]

# Produced by the DVC pipeline; absent on a fresh checkout.
REFERENCE_DATA_PATH = "data/processed/train.csv"


def initialize(app) -> None:
    """Load every default domain's artifacts into the registry."""
    import src.domain_registry as domain_registry
    from src.domain_registry import ensure_domain_initialized

    app.state.model_lock = threading.Lock()
    app.state.model_registry = {}

    for domain_id in DEFAULT_DOMAINS:
        try:
            ensure_domain_initialized(domain_id)
            app.state.model_registry[domain_id] = {
                "model": domain_registry.load_domain_model(domain_id),
                "preprocessor": domain_registry.load_domain_preprocessor(domain_id),
                "version": f"{domain_id}-v1",
            }
        except Exception as e:
            logger.warning(f"Could not load domain model '{domain_id}': {e}")

    # Default telecom container, kept for backward compatibility with callers
    # that predate the per-domain registry.
    telecom = app.state.model_registry.get(
        "telecom", {"model": None, "preprocessor": None, "version": "unknown"}
    )
    app.state.model_container = telecom
    app.state.model = telecom["model"]
    app.state.preprocessor = telecom["preprocessor"]
    app.state.model_version = telecom["version"]
    app.state.model_loaded = telecom["model"] is not None

    # Reference data feeds drift comparison. It is produced by the DVC pipeline,
    # so it is legitimately absent on a fresh checkout or before the first
    # training run — degrade to "no drift baseline" rather than refusing to
    # start. Every domain also carries its own baseline under data/baselines/,
    # which is what drift actually compares against.
    app.state.reference_data = _load_reference_data(REFERENCE_DATA_PATH)


def _load_reference_data(path: str):
    if not Path(path).exists():
        logger.warning(
            "Reference data %s not found; drift comparison will fall back to "
            "each domain's baseline until the training pipeline has run.",
            path,
        )
        return None
    try:
        return pd.read_csv(path).drop(columns=["Churn"], errors="ignore")
    except Exception as exc:
        logger.warning("Could not read reference data %s: %s", path, exc)
        return None


def get_container(app, domain_key: str):
    """Return the registry entry for `domain_key`, or None."""
    with app.state.model_lock:
        return app.state.model_registry.get(domain_key)


def register(app, domain_key: str, model, preprocessor, version: str) -> None:
    with app.state.model_lock:
        app.state.model_registry[domain_key] = {
            "model": model,
            "preprocessor": preprocessor,
            "version": version,
        }


def register_challenger(
    app, domain_key: str, model, preprocessor, version: str
) -> None:
    """Attach a challenger for shadow evaluation; the champion keeps serving."""
    with app.state.model_lock:
        container = app.state.model_registry.setdefault(domain_key, {})
        container["challenger"] = {
            "model": model,
            "preprocessor": preprocessor,
            "version": version,
        }


def promote_challenger(app, domain_key: str) -> str | None:
    """
    Promote a domain's challenger to champion.

    Returns the promoted version, or None when the domain has no challenger.
    """
    with app.state.model_lock:
        container = app.state.model_registry.get(domain_key)
        if not container or "challenger" not in container:
            return None

        challenger = container["challenger"]
        container["champion"] = challenger
        container["model"] = challenger["model"]
        container["preprocessor"] = challenger["preprocessor"]
        container["version"] = challenger["version"]

        # Keep the ONNX engine in step with the model it accelerates, otherwise
        # the promoted champion would be served by the old engine.
        try:
            from src.domain_registry import get_domain_model_dir
            from src.onnx_exporter import ONNXInferenceEngine

            if "onnx_engine" in challenger:
                container["onnx_engine"] = challenger["onnx_engine"]
            else:
                onnx_path = get_domain_model_dir(domain_key) / "model.onnx"
                if Path(onnx_path).exists():
                    container["onnx_engine"] = ONNXInferenceEngine(str(onnx_path))
        except Exception as exc:
            logger.warning(
                "Could not refresh the ONNX engine for domain '%s': %s",
                domain_key,
                exc,
            )

        del container["challenger"]
        return container["version"]


def claim_retraining_slot(app) -> bool:
    """
    Atomically claim the single retraining slot.

    Returns True if this caller moved the status from idle to running and is
    therefore responsible for launching, and eventually clearing, the retrain.
    Checking and setting under the lock is what stops two concurrent drift
    checks from both launching a training run.
    """
    with app.state.model_lock:
        if app.state.retraining_status != "idle":
            return False
        app.state.retraining_status = "running"
        return True


def release_retraining_slot(app) -> None:
    with app.state.model_lock:
        app.state.retraining_status = "idle"


def count_prediction(app, n: int = 1) -> bool:
    """
    Record `n` predictions and report whether a drift check is now due.

    The counter and the boundary test share the lock so a batch that straddles
    the boundary triggers exactly one check.
    """
    with app.state.model_lock:
        previous = app.state.prediction_counter
        app.state.prediction_counter += n
        every = app.state.drift_check_every
        return (app.state.prediction_counter // every) > (previous // every)
