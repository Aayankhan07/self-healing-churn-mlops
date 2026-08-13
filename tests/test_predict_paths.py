"""
Inference path tests.

run_single_prediction resolves its model from one of four places, may route
through an ONNX engine, and may shadow-score a challenger. The HTTP tests only
ever exercise the ordinary path; these cover the rest, including the fallbacks
that exist precisely for when something has gone wrong.
"""

import threading
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from api.database import SessionLocal, ShadowPredictionLog
from api.predict import assign_risk_tier, hash_input, run_single_prediction
from api.schemas import CustomerInput
from src.domain_registry import load_domain_model, load_domain_preprocessor
from src.domains import get_domain_spec
from src.domains.base import RiskBands


@pytest.fixture
def customer(valid_customer):
    return CustomerInput(**valid_customer)


@pytest.fixture
def db():
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="module")
def telecom_artifacts(provision_artifacts):
    return load_domain_model("telecom"), load_domain_preprocessor("telecom")


# ── Risk tiers ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "proba,expected",
    [(0.0, "Low"), (0.34, "Low"), (0.35, "Medium"), (0.64, "Medium"), (0.65, "High")],
)
def test_assign_risk_tier_default_bands(proba, expected):
    assert assign_risk_tier(proba) == expected


def test_assign_risk_tier_uses_the_domain_spec():
    """A domain may band risk differently; the spec wins over the default."""
    spec = SimpleNamespace(risk_bands=RiskBands(low=0.1, high=0.2))
    assert assign_risk_tier(0.15, spec) == "Medium"
    assert assign_risk_tier(0.15) == "Low"


# ── Input hashing ───────────────────────────────────────────


def test_hash_input_is_stable_and_order_independent():
    assert hash_input({"a": 1, "b": 2}) == hash_input({"b": 2, "a": 1})
    assert hash_input({"a": 1}) != hash_input({"a": 2})
    assert len(hash_input({"a": 1})) == 16


# ── Model resolution paths ──────────────────────────────────


def test_resolves_from_the_domain_registry(customer, db, telecom_artifacts):
    model, preprocessor = telecom_artifacts
    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-v7",
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert result.model_version == "telecom-v7"
    assert 0.0 <= result.churn_probability <= 1.0


def test_falls_back_to_the_legacy_model_container(customer, db, telecom_artifacts):
    """Callers predating the per-domain registry still work."""
    model, preprocessor = telecom_artifacts
    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_container={
            "model": model,
            "preprocessor": preprocessor,
            "version": "legacy-v1",
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert result.model_version == "legacy-v1"


def test_falls_back_to_bare_state_attributes(customer, db, telecom_artifacts):
    """The oldest shape: a model hung directly off state, with no lock."""
    model, preprocessor = telecom_artifacts
    state = SimpleNamespace(model=model, preprocessor=preprocessor)

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert result.model_version == "telecom-v1"


def test_loads_from_disk_when_state_has_nothing(customer, db, provision_artifacts):
    """An empty registry must not fail the request — load the artifact."""
    state = SimpleNamespace(model=None, preprocessor=None)

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert 0.0 <= result.churn_probability <= 1.0


# ── ONNX acceleration ───────────────────────────────────────


def test_onnx_engine_is_used_when_present(customer, db, telecom_artifacts):
    model, preprocessor = telecom_artifacts
    engine = MagicMock()
    engine.predict_proba.return_value = [[0.1, 0.9]]

    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-onnx",
                "onnx_engine": engine,
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert engine.predict_proba.called
    assert result.churn_probability == pytest.approx(0.9)
    assert result.risk_tier == "High"


def test_broken_onnx_engine_falls_back_to_the_model(customer, db, telecom_artifacts):
    """Acceleration is an optimization; its failure must not fail the request."""
    model, preprocessor = telecom_artifacts
    engine = MagicMock()
    engine.predict_proba.side_effect = RuntimeError("onnx session died")

    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-onnx",
                "onnx_engine": engine,
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert engine.predict_proba.called
    assert 0.0 <= result.churn_probability <= 1.0


# ── Shadow evaluation ───────────────────────────────────────


def test_challenger_is_shadow_scored_and_logged(customer, db, telecom_artifacts):
    model, preprocessor = telecom_artifacts
    challenger = MagicMock()
    challenger.predict_proba.return_value = [[0.3, 0.7]]

    domain = f"custom_shadow_{uuid.uuid4().hex[:8]}"
    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            domain: {
                "model": model,
                "preprocessor": preprocessor,
                "version": f"{domain}-v1",
                "challenger": {"model": challenger, "version": f"{domain}-v2"},
            }
        },
    )

    before = db.query(ShadowPredictionLog).filter_by(domain_id=domain).count()
    result = run_single_prediction(state, customer, db, domain_id=domain)
    after = db.query(ShadowPredictionLog).filter_by(domain_id=domain).count()

    assert challenger.predict_proba.called
    assert after == before + 1
    # The champion still decides the response.
    assert result.model_version == f"{domain}-v1"


def test_broken_challenger_does_not_fail_the_request(customer, db, telecom_artifacts):
    """Shadow scoring is observation only; it must never affect the caller."""
    model, preprocessor = telecom_artifacts
    challenger = MagicMock()
    challenger.predict_proba.side_effect = RuntimeError("challenger is corrupt")

    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-v1",
                "challenger": {"model": challenger, "version": "telecom-v2"},
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert 0.0 <= result.churn_probability <= 1.0


def test_a_non_dict_challenger_is_ignored(customer, db, telecom_artifacts):
    model, preprocessor = telecom_artifacts
    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-v1",
                "challenger": "not-a-dict",
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert result.model_version == "telecom-v1"


# ── Explanations and enrichment ─────────────────────────────


def test_shap_failure_degrades_to_no_factors(customer, db, telecom_artifacts):
    """An explanation is nice to have; losing it must not lose the prediction."""
    model, _ = telecom_artifacts
    broken_preprocessor = MagicMock()
    broken_preprocessor.transform.side_effect = RuntimeError("shap exploded")

    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": broken_preprocessor,
                "version": "telecom-v1",
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert result.top_factors == []
    assert 0.0 <= result.churn_probability <= 1.0


def test_response_carries_playbook_and_survival(customer, db, telecom_artifacts):
    model, preprocessor = telecom_artifacts
    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-v1",
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert isinstance(result.recommended_actions, list)
    assert result.recommended_actions
    assert result.survival_timeline is not None
    assert result.time_to_churn_days is not None


def test_healed_actions_are_returned_to_the_caller(customer, db, telecom_artifacts):
    model, preprocessor = telecom_artifacts
    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-v1",
            }
        },
    )

    actions = ["Clamped negative tenure to 0"]
    result = run_single_prediction(
        state, customer, db, healed_actions=actions, domain_id="telecom"
    )
    assert result.healed_actions == actions


def test_domain_spec_bands_apply_to_the_response(customer, db, telecom_artifacts):
    """Risk tier comes from the domain's spec, not a global constant."""
    spec = get_domain_spec("telecom")
    model, preprocessor = telecom_artifacts
    state = SimpleNamespace(
        model_lock=threading.Lock(),
        model_registry={
            "telecom": {
                "model": model,
                "preprocessor": preprocessor,
                "version": "telecom-v1",
            }
        },
    )

    result = run_single_prediction(state, customer, db, domain_id="telecom")
    assert result.risk_tier == spec.risk_bands.tier(result.churn_probability)
