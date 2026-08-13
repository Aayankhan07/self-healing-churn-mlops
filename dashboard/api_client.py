"""
The dashboard's single door to the API.

Every HTTP call the UI makes goes through here. Previously thirteen call sites
each built their own request, and several omitted the auth header or the
selected domain — so `/health` and `/drift/report` reported on telecom no
matter which domain the user had chosen, and would have started failing the
moment those routes required a key.
"""

import logging
import os

import requests
import streamlit as st

logger = logging.getLogger(__name__)

API_URL = os.getenv("API_URL", "http://localhost:8000")

ENVIRONMENT = os.getenv("ENV", "development").lower()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    if ENVIRONMENT == "production":
        raise ValueError(
            "API_KEY environment variable must be set in production environment!"
        )
    API_KEY = "dev-key-change-in-prod"

HEADERS = {"X-API-Key": API_KEY}

DEFAULT_TIMEOUT = 5


def _get(path, domain=None, timeout=DEFAULT_TIMEOUT, **params):
    if domain is not None:
        params["domain"] = domain
    return requests.get(
        f"{API_URL}{path}", headers=HEADERS, params=params, timeout=timeout
    )


def _post(path, domain=None, timeout=DEFAULT_TIMEOUT, **kwargs):
    params = kwargs.pop("params", {})
    if domain is not None:
        params["domain"] = domain
    return requests.post(
        f"{API_URL}{path}", headers=HEADERS, params=params, timeout=timeout, **kwargs
    )


# ── Reads ───────────────────────────────────────────────────


@st.cache_data(ttl=15)
def get_health(domain=None):
    """Service health for a domain. Degrades quietly — the UI shows a banner."""
    try:
        return _get("/health", domain=domain, timeout=3).json()
    except Exception as exc:
        logger.debug("health check failed: %s", exc)
        return {
            "status": "degraded",
            "model_loaded": False,
            "model_version": "unknown",
            "demo_fixture": False,
        }


@st.cache_data(ttl=30)
def get_drift_status(domain=None):
    try:
        return _get("/drift/status", domain=domain, timeout=3).json()
    except Exception as exc:
        logger.debug("drift status failed: %s", exc)
        return {"status": "healthy", "drift_detected": False}


@st.cache_data(ttl=30)
def get_metrics(domain=None):
    try:
        return _get("/metrics", domain=domain).json()
    except Exception as exc:
        logger.debug("metrics failed: %s", exc)
        return None


@st.cache_data(ttl=15)
def get_self_healing_logs(domain=None, limit=100):
    try:
        return _get("/self-healing/logs", domain=domain, limit=limit).json()
    except Exception as exc:
        logger.debug("self-healing logs failed: %s", exc)
        return []


@st.cache_data(ttl=30)
def get_shadow_status(domain=None):
    try:
        return _get("/model/shadow-status", domain=domain).json()
    except Exception as exc:
        logger.debug("shadow status failed: %s", exc)
        return None


def get_drift_report(domain=None):
    """Raw HTML of the latest drift report, or None."""
    try:
        r = _get("/drift/report", domain=domain)
        return r.text if r.status_code == 200 else None
    except Exception as exc:
        logger.debug("drift report failed: %s", exc)
        return None


# ── Writes ──────────────────────────────────────────────────


def predict(customer, domain=None, timeout=10):
    return _post("/predict", domain=domain, json=customer, timeout=timeout)


def predict_batch(customers, domain=None, timeout=60):
    return _post(
        "/predict/batch", domain=domain, json={"customers": customers}, timeout=timeout
    )


def upload_csv(filename, content, domain=None, timeout=120):
    return _post(
        "/upload",
        domain=domain,
        files={"file": (filename, content, "text/csv")},
        timeout=timeout,
    )


def trigger_retrain(domain=None, timeout=10):
    return _post("/self-healing/trigger-retrain", domain=domain, timeout=timeout)


def promote_model(domain=None, timeout=10):
    return _post("/model/promote", domain=domain, timeout=timeout)


def bootstrap_domain(domain_name, timeout=30):
    return _post(
        "/domain/bootstrap", json={"domain_name": domain_name}, timeout=timeout
    )
