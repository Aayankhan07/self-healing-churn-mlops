"""
Model registry tests.

This module is the only code permitted to touch app.state.model_registry, and
it must always do so under the lock. Its promotion and slot-claiming logic is
what stops a challenger being swapped out from under an in-flight request, so
it is worth testing directly rather than only through the routes.
"""

import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from api.services import model_registry


@pytest.fixture
def app():
    return SimpleNamespace(
        state=SimpleNamespace(
            model_lock=threading.Lock(),
            model_registry={},
            retraining_status="idle",
            prediction_counter=0,
            drift_check_every=100,
        )
    )


# ── Registration ────────────────────────────────────────────


def test_register_and_read_back(app):
    model, preprocessor = MagicMock(), MagicMock()
    model_registry.register(app, "telecom", model, preprocessor, "telecom-v1")

    container = model_registry.get_container(app, "telecom")
    assert container["model"] is model
    assert container["preprocessor"] is preprocessor
    assert container["version"] == "telecom-v1"


def test_get_container_for_an_unknown_domain(app):
    assert model_registry.get_container(app, "nope") is None


def test_register_replaces_an_existing_entry(app):
    first, second = MagicMock(), MagicMock()
    model_registry.register(app, "telecom", first, MagicMock(), "v1")
    model_registry.register(app, "telecom", second, MagicMock(), "v2")

    container = model_registry.get_container(app, "telecom")
    assert container["model"] is second
    assert container["version"] == "v2"


# ── Challenger lifecycle ────────────────────────────────────


def test_register_challenger_leaves_the_champion_serving(app):
    champion, challenger = MagicMock(), MagicMock()
    model_registry.register(app, "telecom", champion, MagicMock(), "v1")
    model_registry.register_challenger(app, "telecom", challenger, MagicMock(), "v2")

    container = model_registry.get_container(app, "telecom")
    assert container["model"] is champion, "champion must keep serving"
    assert container["challenger"]["model"] is challenger


def test_register_challenger_for_an_unseen_domain(app):
    """setdefault path: a domain whose champion was never registered."""
    model_registry.register_challenger(app, "brand_new", MagicMock(), MagicMock(), "v2")
    assert "challenger" in model_registry.get_container(app, "brand_new")


def test_promote_swaps_challenger_into_champion(app):
    champion, challenger = MagicMock(), MagicMock()
    model_registry.register(app, "telecom", champion, MagicMock(), "v1")
    model_registry.register_challenger(app, "telecom", challenger, MagicMock(), "v2")

    promoted = model_registry.promote_challenger(app, "telecom")

    assert promoted == "v2"
    container = model_registry.get_container(app, "telecom")
    assert container["model"] is challenger
    assert container["version"] == "v2"
    assert "challenger" not in container, "challenger is consumed by promotion"


def test_promote_without_a_challenger_returns_none(app):
    model_registry.register(app, "telecom", MagicMock(), MagicMock(), "v1")
    assert model_registry.promote_challenger(app, "telecom") is None


def test_promote_an_unknown_domain_returns_none(app):
    assert model_registry.promote_challenger(app, "nope") is None


def test_promotion_carries_the_challengers_onnx_engine(app):
    engine = MagicMock()
    model_registry.register(app, "telecom", MagicMock(), MagicMock(), "v1")
    with app.state.model_lock:
        app.state.model_registry["telecom"]["challenger"] = {
            "model": MagicMock(),
            "preprocessor": MagicMock(),
            "version": "v2",
            "onnx_engine": engine,
        }

    model_registry.promote_challenger(app, "telecom")
    assert model_registry.get_container(app, "telecom")["onnx_engine"] is engine


def test_promotion_survives_an_onnx_failure(app):
    """A broken accelerator must not block the promotion itself."""
    model_registry.register(app, "telecom", MagicMock(), MagicMock(), "v1")
    model_registry.register_challenger(app, "telecom", MagicMock(), MagicMock(), "v2")

    with patch(
        "src.domain_registry.get_domain_model_dir",
        side_effect=RuntimeError("disk gone"),
    ):
        assert model_registry.promote_challenger(app, "telecom") == "v2"


# ── Retraining slot ─────────────────────────────────────────


def test_claim_then_release(app):
    assert model_registry.claim_retraining_slot(app) is True
    assert app.state.retraining_status == "running"
    assert model_registry.claim_retraining_slot(app) is False

    model_registry.release_retraining_slot(app)
    assert app.state.retraining_status == "idle"
    assert model_registry.claim_retraining_slot(app) is True


def test_exactly_one_of_many_threads_claims_the_slot(app):
    claims = []
    barrier = threading.Barrier(12)

    def worker():
        barrier.wait()
        claims.append(model_registry.claim_retraining_slot(app))

    threads = [threading.Thread(target=worker) for _ in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(claims) == 1


# ── Prediction counting ─────────────────────────────────────


def test_counter_signals_a_drift_check_on_the_boundary(app):
    app.state.drift_check_every = 5

    signals = [model_registry.count_prediction(app) for _ in range(10)]

    assert app.state.prediction_counter == 10
    # Exactly two boundaries crossed in ten predictions.
    assert sum(signals) == 2
    assert signals[4] is True and signals[9] is True


def test_a_batch_straddling_the_boundary_signals_once(app):
    """The bug this guards: a batch used to trigger a check per boundary."""
    app.state.drift_check_every = 100
    assert model_registry.count_prediction(app, 250) is True
    assert app.state.prediction_counter == 250


def test_counting_below_the_boundary_does_not_signal(app):
    app.state.drift_check_every = 100
    assert model_registry.count_prediction(app, 99) is False


def test_concurrent_counting_loses_no_predictions(app):
    app.state.drift_check_every = 1_000_000  # never trip the boundary
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        for _ in range(50):
            model_registry.count_prediction(app)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert app.state.prediction_counter == 400


# ── Reference data ──────────────────────────────────────────


def test_missing_reference_data_is_tolerated(tmp_path):
    """A fresh checkout has no training split; startup must not depend on it."""
    assert model_registry._load_reference_data(str(tmp_path / "absent.csv")) is None


def test_unreadable_reference_data_is_tolerated(tmp_path):
    """A read failure degrades to no baseline rather than killing startup."""
    path = tmp_path / "locked.csv"
    path.write_text("tenure\n1\n")

    with patch("pandas.read_csv", side_effect=PermissionError("file is locked")):
        assert model_registry._load_reference_data(str(path)) is None


def test_reference_data_drops_the_target_column(tmp_path):
    import pandas as pd

    path = tmp_path / "train.csv"
    pd.DataFrame({"tenure": [1, 2], "Churn": [0, 1]}).to_csv(path, index=False)

    frame = model_registry._load_reference_data(str(path))
    assert "Churn" not in frame.columns
    assert "tenure" in frame.columns
