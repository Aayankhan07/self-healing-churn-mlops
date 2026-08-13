"""
Migration tests.

The schema drifted from the migration history once already: init_db() added
the domain_id columns at runtime with a hand-rolled ALTER TABLE, so
`alembic upgrade head` on a fresh database produced a schema the application
could not actually use. These tests make that failure mode loud.
"""

import sqlite3

import pytest
from alembic import command
from alembic.config import Config

from api.database import Base

# Tables the application reads and writes, with the columns it depends on.
DOMAIN_SCOPED_TABLES = ["predictions", "self_healing_logs", "drift_reports"]


def _alembic_config(db_path):
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return config


@pytest.fixture
def migrated_db(tmp_path):
    """A database built only by `alembic upgrade head` — no init_db()."""
    db_path = tmp_path / "migrated.db"
    command.upgrade(_alembic_config(db_path), "head")
    return db_path


def _columns(db_path, table):
    conn = sqlite3.connect(db_path)
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    finally:
        conn.close()


def _tables(db_path):
    conn = sqlite3.connect(db_path)
    try:
        return {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def test_migrations_alone_produce_every_table(migrated_db):
    """A fresh deploy runs migrations, not init_db()."""
    present = _tables(migrated_db)
    for table in Base.metadata.tables:
        assert table in present, f"{table} is missing from the migration history"


def test_migrations_match_the_orm_columns(migrated_db):
    """
    Every column the ORM declares must exist after migrating. This is the
    check that would have caught domain_id being added only at runtime.
    """
    mismatches = {}
    for table_name, table in Base.metadata.tables.items():
        expected = {c.name for c in table.columns}
        actual = _columns(migrated_db, table_name)
        missing = expected - actual
        if missing:
            mismatches[table_name] = missing
    assert not mismatches, f"migrations do not create these columns: {mismatches}"


def test_domain_id_is_indexed(migrated_db):
    """Every domain-scoped query filters on it."""
    conn = sqlite3.connect(migrated_db)
    try:
        indexes = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
    finally:
        conn.close()

    for table in DOMAIN_SCOPED_TABLES + ["shadow_predictions"]:
        assert f"ix_{table}_domain_id" in indexes


def test_upgrade_is_idempotent_over_a_runtime_patched_database(tmp_path):
    """
    Existing databases already carry domain_id, courtesy of the old init_db().
    Migrating one must succeed rather than fail on a duplicate column.
    """
    db_path = tmp_path / "legacy.db"
    config = _alembic_config(db_path)

    # Build the pre-domain schema, then patch it the way init_db() used to.
    command.upgrade(config, "10033d7c865c")
    conn = sqlite3.connect(db_path)
    try:
        for table in DOMAIN_SCOPED_TABLES:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN domain_id VARCHAR DEFAULT 'telecom'"
            )
        conn.execute(
            "INSERT INTO predictions "
            "(id, customer_id, input_hash, probability, risk_tier, prediction, model_ver) "
            "VALUES ('legacy-1', 'C1', 'h', 0.5, 'Medium', 0, 'v1')"
        )
        conn.commit()
    finally:
        conn.close()

    command.upgrade(config, "head")

    conn = sqlite3.connect(db_path)
    try:
        # The pre-existing row survives and is attributed to the original domain.
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 1
        assert (
            conn.execute("SELECT domain_id FROM predictions").fetchone()[0] == "telecom"
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE domain_id IS NULL"
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_init_db_no_longer_alters_tables():
    """Schema changes belong in migrations, not in application startup."""
    import inspect

    import api.database as database

    source = inspect.getsource(database.init_db)
    # Strip the docstring: it explains the old behavior and would match itself.
    body = source.split('"""')[-1]
    assert "ALTER TABLE" not in body.upper()
