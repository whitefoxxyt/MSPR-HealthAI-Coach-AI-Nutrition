from __future__ import annotations

import os
import re
from collections.abc import Callable, Generator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import respx
from jose import jwt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

MIGRATIONS_DIR = Path(os.environ.get("MIGRATIONS_DIR", "/migrations"))
MIGRATION_PATTERN = re.compile(r"^V(\d+)__.*\.sql$")
MAX_MIGRATION_VERSION = 11  # PRD #45 slice 1 : V11 ajoute imbalances, serving_sizes, meal_type, compliance_status, compliance_warnings

TEST_AUTH_SECRET = "test-secret-not-for-prod-do-not-use"
TEST_OLLAMA_HOST = "http://ollama-test:11434"

# Tables truncatees avant chaque test (isolation function-scope).
# nutrition_entries vient de l'ETL, le reste est cree en V8 puis enrichi V9/V10/V11.
_TRUNCATE_TABLES = [
    "meal_analyses",
    "meal_plans",
    "nutrition_goals",
    "nutrition_entries",
]


def _ordered_migrations() -> list[Path]:
    files: list[tuple[int, Path]] = []
    for path in MIGRATIONS_DIR.iterdir():
        match = MIGRATION_PATTERN.match(path.name)
        if match:
            version = int(match.group(1))
            if version <= MAX_MIGRATION_VERSION:
                files.append((version, path))
    return [p for _, p in sorted(files)]


@pytest.fixture(scope="session")
def pg_container() -> Generator[PostgresContainer, None, None]:
    """Spawn un PostgreSQL 17 ephemere et joue les migrations jusqu'a MAX_MIGRATION_VERSION."""
    container = PostgresContainer("postgres:17-alpine", driver="psycopg2")
    container.start()
    try:
        engine = create_engine(container.get_connection_url(), pool_pre_ping=True)
        with engine.begin() as conn:
            for migration in _ordered_migrations():
                conn.execute(text(migration.read_text()))
        engine.dispose()
        yield container
    finally:
        container.stop()


@pytest.fixture(scope="session")
def pg_engine(pg_container: PostgresContainer) -> Generator[Engine, None, None]:
    engine = create_engine(pg_container.get_connection_url(), pool_pre_ping=True)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def db_session(pg_engine: Engine) -> Generator[Session, None, None]:
    """Session SQLAlchemy isolee : truncate des tables MSPR2 avant chaque test."""
    SessionLocal = sessionmaker(bind=pg_engine)
    session = SessionLocal()
    try:
        tables = ", ".join(_TRUNCATE_TABLES)
        session.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
        session.commit()
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_ollama() -> Generator[respx.MockRouter, None, None]:
    """Intercepte les appels httpx vers Ollama via respx.

    Reponse JSON par defaut sur POST /api/generate ; le test peut surcharger
    via mock_ollama.post(...).respond(...) avant l'appel.
    """
    with respx.mock(assert_all_called=False) as router:
        router.post(re.compile(r".*/api/generate$")).respond(
            200,
            json={"response": "{}", "done": True},
        )
        router.get(re.compile(r".*/api/tags$")).respond(
            200,
            json={"models": [{"name": "gemma3:4b"}]},
        )
        yield router


@pytest.fixture
def mock_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[..., list[dict[str, Any]]]:
    """Patch food_classifier._get_classifier pour retourner un fake deterministe.

    Le fake renvoie pizza (0.85) puis lasagna (0.07), tronque a top_k.
    """
    from app.services import food_classifier

    def fake_pipeline(image: Any, top_k: int = 5) -> list[dict[str, Any]]:
        results = [
            {"label": "pizza", "score": 0.85},
            {"label": "lasagna", "score": 0.07},
            {"label": "spaghetti_bolognese", "score": 0.03},
        ]
        return results[:top_k]

    monkeypatch.setattr(food_classifier, "_get_classifier", lambda: fake_pipeline)
    return fake_pipeline


@pytest.fixture
def valid_jwt() -> Callable[..., str]:
    """Fabrique un JWT HS256 signe avec TEST_AUTH_SECRET.

    valid_jwt(user_id=42, email="x@y.z", expires_in=3600) -> str
    """

    def _make(
        user_id: int = 1,
        email: str | None = "test@example.com",
        expires_in: int = 3600,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        now = datetime.now(timezone.utc)
        payload: dict[str, Any] = {
            "sub": str(user_id),
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=expires_in)).timestamp()),
        }
        if email is not None:
            payload["email"] = email
        if extra_claims:
            payload.update(extra_claims)
        return jwt.encode(payload, TEST_AUTH_SECRET, algorithm="HS256")

    return _make
