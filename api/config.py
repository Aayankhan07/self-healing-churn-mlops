"""
Runtime configuration.

One place to see every environment variable the service reads, with its default
and its meaning. Values are resolved at import; nothing writes to os.environ.

Secrets are deliberately not here — API keys live in api/security.py, which
enforces that production supplies each one explicitly.
"""

import os
from dataclasses import dataclass


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}")


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}")


@dataclass(frozen=True)
class Settings:
    """Resolved service configuration."""

    # Where the operational store lives.
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./churnguard.db")

    # MLflow tracking backend.
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

    # Run a drift check every N predictions.
    drift_check_every: int = _int_env("DRIFT_CHECK_EVERY_N", 100)

    # Share of drifted columns above which retraining is triggered.
    drift_threshold: float = _float_env("DRIFT_THRESHOLD", 0.20)

    # Optional Slack webhook for retraining and promotion alerts.
    slack_webhook_url: str = os.getenv("SLACK_WEBHOOK_URL", "")

    @property
    def slack_enabled(self) -> bool:
        return bool(self.slack_webhook_url)


settings = Settings()
