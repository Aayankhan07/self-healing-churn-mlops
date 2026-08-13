"""
Shared FastAPI dependencies.

Routers declare what they need — a database session, a scope — and this module
supplies it.
"""

from fastapi import Header, HTTPException

from api.database import get_db  # noqa: F401  (re-exported for routers)
from api.security import scopes_for


def verify_scope(required_scope: str):
    """Dependency factory verifying an API key carries `required_scope`."""

    def scope_checker(x_api_key: str = Header(...)):
        scopes = scopes_for(x_api_key)
        if scopes is None:
            raise HTTPException(status_code=403, detail="Invalid API key")

        if required_scope not in scopes:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Forbidden: Insufficient privileges. "
                    f"Required scope '{required_scope}'."
                ),
            )
        return x_api_key

    return scope_checker


def verify_api_key(x_api_key: str = Header(...)):
    """Backwards-compatible dependency for read:predict scope."""
    return verify_scope("read:predict")(x_api_key)
