from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

import sys
import os
from pathlib import Path

# Alembic runs this file as a script from its own directory, not as part of the
# installed package, so the project root has to go on the path explicitly even
# though `pip install -e .` covers every other entry point.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from api.database import Base  # noqa: E402

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Resolve the database URL, in precedence order:
#   1. one set programmatically on the Config (tests, tooling)
#   2. DATABASE_URL from the environment (deploys)
#   3. the local development default
# Checking the Config first matters: overwriting it unconditionally meant a
# caller that passed an explicit URL silently migrated whatever DATABASE_URL
# happened to point at instead.
configured_url = config.get_main_option("sqlalchemy.url", None)
PLACEHOLDER_URL = "driver://user:pass@localhost/dbname"

if not configured_url or configured_url == PLACEHOLDER_URL:
    configured_url = os.getenv("DATABASE_URL", "sqlite:///./churnguard.db")

config.set_main_option("sqlalchemy.url", configured_url)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
