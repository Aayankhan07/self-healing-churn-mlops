"""
API key handling and role-based scopes.

Keys are compared by SHA-256 digest rather than in plaintext. Each role maps to
the scopes it may exercise; routes declare the scope they require and the
dependency in api/dependencies.py enforces it.
"""

import hashlib
import os
import secrets

ENVIRONMENT = os.getenv("ENV", "development").lower()
IS_PRODUCTION = ENVIRONMENT == "production"

ANALYST_SCOPES = ["read:predict"]
ENGINEER_SCOPES = ["read:predict", "write:retrain"]
ADMIN_SCOPES = ["read:predict", "write:retrain", "admin:bootstrap", "admin:promote"]

# Every role key must be supplied explicitly in production. Falling back to a
# shared or well-known default would silently grant admin scope to whoever
# holds the read key.
_DEV_DEFAULTS = {
    "API_KEY": "dev-key-change-in-prod",
    "API_KEY_ANALYST": "analyst-key",
    "API_KEY_ENGINEER": "engineer-key",
}


def _require_key(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value
    if IS_PRODUCTION:
        raise ValueError(
            f"{name} environment variable must be set in production environment!"
        )
    return _DEV_DEFAULTS[name]


API_KEY = _require_key("API_KEY")
API_KEY_ANALYST = _require_key("API_KEY_ANALYST")
API_KEY_ENGINEER = _require_key("API_KEY_ENGINEER")

API_KEY_ADMIN = os.getenv("API_KEY_ADMIN")
if not API_KEY_ADMIN:
    if IS_PRODUCTION:
        raise ValueError(
            "API_KEY_ADMIN environment variable must be set in production environment!"
        )
    API_KEY_ADMIN = API_KEY


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def generate_high_entropy_key(prefix: str = "cg") -> str:
    """Utility generating 256-bit cryptographically secure high-entropy API token."""
    return f"{prefix}_{secrets.token_urlsafe(32)}"


# Hashed Role-Based Access Control (RBAC) Scope Mappings
HASHED_API_KEY_SCOPES = {
    _hash_key(API_KEY_ANALYST): ANALYST_SCOPES,
    _hash_key(API_KEY_ENGINEER): ENGINEER_SCOPES,
    _hash_key(API_KEY_ADMIN): ADMIN_SCOPES,
}


def scopes_for(x_api_key: str):
    """
    Return the scopes granted to `x_api_key`, or None if it is not recognized.
    """
    scopes = HASHED_API_KEY_SCOPES.get(_hash_key(x_api_key))
    if scopes:
        return scopes

    # Legacy single-key fallback. Outside production only: granting the full
    # admin scope to the generic API_KEY is exactly the escalation path we
    # refuse to ship.
    if not IS_PRODUCTION and secrets.compare_digest(
        _hash_key(x_api_key), _hash_key(API_KEY)
    ):
        return ADMIN_SCOPES
    return None
