"""
Security regression tests for the RBAC key handling.

These run api.main in a subprocess because the key configuration is resolved at
import time; reimporting in-process would leak state across tests.
"""

import subprocess
import sys
import textwrap

PROD_ENV = {
    "ENV": "production",
    "API_KEY": "prod-generic",
    "API_KEY_ANALYST": "prod-analyst",
    "API_KEY_ENGINEER": "prod-engineer",
    "API_KEY_ADMIN": "prod-admin",
}


def _run(script, env):
    import os

    full_env = {**os.environ, **env}
    # Ensure the missing-key cases really are missing.
    for key in [
        "API_KEY",
        "API_KEY_ANALYST",
        "API_KEY_ENGINEER",
        "API_KEY_ADMIN",
    ]:
        if key not in env:
            full_env.pop(key, None)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        env=full_env,
    )


def test_production_requires_every_role_key():
    """Booting production without explicit keys must fail loudly, not default."""
    for omitted in [
        "API_KEY",
        "API_KEY_ANALYST",
        "API_KEY_ENGINEER",
        "API_KEY_ADMIN",
    ]:
        env = {k: v for k, v in PROD_ENV.items() if k != omitted}
        result = _run("import api.main", env)
        assert result.returncode != 0, f"{omitted} missing should abort startup"
        assert omitted in result.stderr
        assert "must be set in production" in result.stderr


def test_production_boots_with_all_keys():
    result = _run(
        """
        import api.main as m
        assert m.IS_PRODUCTION is True
        assert len(m.HASHED_API_KEY_SCOPES) == 3
        print("ok")
        """,
        PROD_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_generic_key_does_not_grant_admin_in_production():
    """
    The legacy fallback granted full admin scope to whoever presented API_KEY.
    In production that turned a leaked read key into an admin key.
    """
    result = _run(
        """
        from fastapi import HTTPException
        import api.main as m

        try:
            m.verify_scope("admin:promote")(x_api_key="prod-generic")
        except HTTPException as exc:
            print("rejected", exc.status_code)
        else:
            print("ESCALATED")
        """,
        PROD_ENV,
    )
    assert result.returncode == 0, result.stderr
    assert "rejected 403" in result.stdout
    assert "ESCALATED" not in result.stdout


def test_analyst_key_never_reaches_admin_scope():
    result = _run(
        """
        from fastapi import HTTPException
        import api.main as m

        try:
            m.verify_scope("admin:promote")(x_api_key="prod-analyst")
        except HTTPException as exc:
            print("rejected", exc.status_code)
        else:
            print("ESCALATED")
        """,
        PROD_ENV,
    )
    assert "rejected 403" in result.stdout
    assert "ESCALATED" not in result.stdout


def test_dev_mode_keeps_legacy_single_key_fallback():
    """Outside production the generic key still works, so local setups don't break."""
    result = _run(
        """
        import api.main as m
        assert m.verify_scope("admin:promote")(
            x_api_key="dev-key-change-in-prod"
        ) == "dev-key-change-in-prod"
        print("ok")
        """,
        {"ENV": "development"},
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
