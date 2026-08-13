"""add domain_id columns and the shadow_predictions table

Brings the migration history in line with the schema the application actually
uses. Both changes previously happened at runtime in init_db(), which meant a
fresh deploy running `alembic upgrade head` produced a schema missing the
domain_id columns and the shadow_predictions table entirely — the application
only worked because init_db() patched it on every boot.

Written to be idempotent: existing databases already carry these objects
courtesy of init_db(), so each step checks before acting rather than failing on
a duplicate column.

Revision ID: 2b91f4c7a3de
Revises: 10033d7c865c
Create Date: 2026-08-13

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "2b91f4c7a3de"
down_revision: Union[str, None] = "10033d7c865c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Tables that gained a domain_id when the platform became multi-tenant.
DOMAIN_TABLES = ("predictions", "self_healing_logs", "drift_reports")

# Every pre-existing row was written before domains existed, so it belongs to
# the original telecom domain.
LEGACY_DOMAIN = "telecom"


def _existing_columns(table: str) -> set:
    inspector = sa.inspect(op.get_bind())
    return {col["name"] for col in inspector.get_columns(table)}


def _table_exists(table: str) -> bool:
    return table in sa.inspect(op.get_bind()).get_table_names()


def upgrade() -> None:
    for table in DOMAIN_TABLES:
        if not _table_exists(table):
            continue
        if "domain_id" not in _existing_columns(table):
            op.add_column(
                table,
                sa.Column(
                    "domain_id",
                    sa.String(),
                    nullable=True,
                    server_default=LEGACY_DOMAIN,
                ),
            )
        # Backfill rows that predate the column, including any the runtime
        # ALTER TABLE added without a value.
        op.execute(
            sa.text(
                f"UPDATE {table} SET domain_id = :domain WHERE domain_id IS NULL"
            ).bindparams(domain=LEGACY_DOMAIN)
        )

        existing_indexes = {
            ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes(table)
        }
        index_name = f"ix_{table}_domain_id"
        if index_name not in existing_indexes:
            # Every domain-scoped query filters on this column.
            op.create_index(index_name, table, ["domain_id"])

    if not _table_exists("shadow_predictions"):
        op.create_table(
            "shadow_predictions",
            sa.Column("id", sa.String(), nullable=False),
            sa.Column(
                "domain_id", sa.String(), nullable=True, server_default=LEGACY_DOMAIN
            ),
            sa.Column("customer_id", sa.String(), nullable=True),
            sa.Column("champion_proba", sa.Float(), nullable=False),
            sa.Column("challenger_proba", sa.Float(), nullable=False),
            sa.Column("probability_delta", sa.Float(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=True),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(
            "ix_shadow_predictions_domain_id", "shadow_predictions", ["domain_id"]
        )


def downgrade() -> None:
    if _table_exists("shadow_predictions"):
        op.drop_index("ix_shadow_predictions_domain_id", "shadow_predictions")
        op.drop_table("shadow_predictions")

    for table in DOMAIN_TABLES:
        if not _table_exists(table):
            continue
        existing_indexes = {
            ix["name"] for ix in sa.inspect(op.get_bind()).get_indexes(table)
        }
        index_name = f"ix_{table}_domain_id"
        if index_name in existing_indexes:
            op.drop_index(index_name, table)
        if "domain_id" in _existing_columns(table):
            op.drop_column(table, "domain_id")
