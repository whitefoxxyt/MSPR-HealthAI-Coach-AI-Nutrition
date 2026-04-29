from __future__ import annotations

import os
import re
from collections.abc import Generator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

MIGRATIONS_DIR = Path(os.environ.get("MIGRATIONS_DIR", "/migrations"))
MIGRATION_PATTERN = re.compile(r"^V(\d+)__.*\.sql$")
MAX_VERSION = 8  # AC issue 18 : V1 a V8


def _ordered_migrations() -> list[Path]:
    files = []
    for path in MIGRATIONS_DIR.iterdir():
        match = MIGRATION_PATTERN.match(path.name)
        if match:
            version = int(match.group(1))
            if version <= MAX_VERSION:
                files.append((version, path))
    return [p for _, p in sorted(files)]


@pytest.fixture(scope="session")
def pg_engine():
    """Spawn un PG17, joue V1 a V8, retourne l'engine SQLAlchemy."""
    container = PostgresContainer("postgres:17-alpine", driver="psycopg2")
    container.start()
    try:
        engine = create_engine(container.get_connection_url(), pool_pre_ping=True)
        with engine.begin() as conn:
            for migration in _ordered_migrations():
                conn.execute(text(migration.read_text()))
        yield engine
        engine.dispose()
    finally:
        container.stop()


@pytest.fixture
def db_session(pg_engine) -> Generator[Session, None, None]:
    """Session SQLAlchemy isolee : truncate nutrition_entries avant chaque test."""
    SessionLocal = sessionmaker(bind=pg_engine)
    session = SessionLocal()
    try:
        session.execute(text("TRUNCATE TABLE nutrition_entries RESTART IDENTITY CASCADE"))
        session.commit()
        yield session
    finally:
        session.close()
