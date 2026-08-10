"""
Startup resilience tests.

The API loads four domain models at startup and is written to degrade
gracefully when one is unavailable — a missing or corrupt artifact should log a
warning and leave the remaining domains serving, not abort the process.
"""

import threading

import pytest
from fastapi.testclient import TestClient


def test_startup_survives_domain_model_load_failure(monkeypatch):
    """
    Injecting a load failure must not prevent the app from starting.

    Phase 1 bug #1: the failure path calls `logger.warning(...)` but `logger` is
    never defined in api.main, so the handler that exists to swallow the error
    raises NameError and kills startup instead.
    """
    import api.main as main

    def boom(domain_id):
        raise RuntimeError(f"artifact missing for {domain_id}")

    monkeypatch.setattr("src.domain_registry.load_domain_model", boom)

    with TestClient(main.app) as client:
        assert client.get("/health").status_code == 200


def test_startup_initializes_state():
    """Startup must publish the state later requests depend on."""
    import api.main as main

    with TestClient(main.app) as client:
        client.get("/health")
        state = main.app.state
        assert isinstance(state.model_lock, threading.Lock().__class__)
        assert isinstance(state.model_registry, dict)
        assert state.retraining_status == "idle"
        assert state.prediction_counter >= 0
        assert state.drift_check_every > 0


@pytest.mark.parametrize("domain", ["telecom", "school", "ecommerce", "fitness"])
def test_default_domains_registered(domain):
    import api.main as main

    with TestClient(main.app) as client:
        client.get("/health")
        assert domain in main.app.state.model_registry
