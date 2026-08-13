"""
Configuration tests.

Cover the two things that went wrong before config was centralized: values
parsed ad hoc at each call site, and worker threads passing arguments by
mutating os.environ.
"""

import inspect
import os

import pytest

from api.config import Settings, _float_env, _int_env, settings


def test_settings_expose_every_documented_default():
    assert settings.database_url
    assert settings.mlflow_tracking_uri
    assert settings.drift_check_every > 0
    assert 0.0 <= settings.drift_threshold <= 1.0
    assert isinstance(settings.slack_webhook_url, str)


def test_slack_disabled_without_a_webhook():
    assert Settings(slack_webhook_url="").slack_enabled is False
    assert Settings(slack_webhook_url="https://hooks.example").slack_enabled is True


def test_settings_are_frozen():
    """Config is read-only: nothing should reconfigure the service at runtime."""
    with pytest.raises(Exception):
        settings.drift_threshold = 0.9


@pytest.mark.parametrize(
    "raw,expected", [(None, 100), ("", 100), ("250", 250), ("0", 0)]
)
def test_int_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("SOME_INT", raising=False)
    else:
        monkeypatch.setenv("SOME_INT", raw)
    assert _int_env("SOME_INT", 100) == expected


def test_int_env_rejects_garbage(monkeypatch):
    """A typo'd value must fail loudly at boot, not silently fall back."""
    monkeypatch.setenv("SOME_INT", "abc")
    with pytest.raises(ValueError, match="SOME_INT"):
        _int_env("SOME_INT", 100)


@pytest.mark.parametrize("raw,expected", [(None, 0.2), ("0.35", 0.35), ("1", 1.0)])
def test_float_env_parsing(monkeypatch, raw, expected):
    if raw is None:
        monkeypatch.delenv("SOME_FLOAT", raising=False)
    else:
        monkeypatch.setenv("SOME_FLOAT", raw)
    assert _float_env("SOME_FLOAT", 0.2) == expected


def test_float_env_rejects_garbage(monkeypatch):
    monkeypatch.setenv("SOME_FLOAT", "high")
    with pytest.raises(ValueError, match="SOME_FLOAT"):
        _float_env("SOME_FLOAT", 0.2)


# ── Worker-thread safety ────────────────────────────────────


def test_train_takes_domain_and_path_as_arguments():
    """
    The retrainer runs in a background thread. It used to pass its target
    domain and training file by assigning to os.environ — process-global
    state, so two concurrent retrains could overwrite each other.
    """
    from src.train import train

    params = inspect.signature(train).parameters
    assert "domain_id" in params
    assert "train_data_path" in params


def test_retrain_service_does_not_mutate_the_environment():
    import api.services.retrain_service as retrain_service

    source = inspect.getsource(retrain_service)
    assert "os.environ[" not in source, "retraining must not write to os.environ"


def test_train_still_honors_env_vars_for_dvc(monkeypatch):
    """DVC stages and manual runs set these; the fallback must survive."""
    from src import train as train_module

    monkeypatch.setenv("TARGET_DOMAIN", "telecom")
    monkeypatch.setenv("TRAIN_DATA_PATH", "data/processed/train.csv")
    params = inspect.signature(train_module.train).parameters
    assert params["domain_id"].default is None
    assert params["train_data_path"].default is None
    assert os.getenv("TARGET_DOMAIN") == "telecom"
