from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from testcontainers.postgres import PostgresContainer

from app.config import settings
from app.models.schemas import (
    HealthGoal,
    ImbalanceStatus,
    ImbalanceTag,
    Nutrient,
    PlanInputs,
)
from app.services.llm_client import generate_plan, generate_recommendation


@pytest.fixture(scope="module")
def ollama_host() -> str:
    host = os.environ.get("OLLAMA_HOST")
    if not host:
        pytest.skip("OLLAMA_HOST not set, skipping real Ollama integration tests.")
    try:
        httpx.get(f"{host}/api/tags", timeout=5.0).raise_for_status()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        pytest.skip(f"Ollama unreachable at {host}: {exc}")
    return host


@pytest.fixture(scope="module")
def real_db_session(ollama_host: str) -> Generator[Session, None, None]:
    """Spawn un PG17, joue les migrations V1-V9, retourne une session."""
    migrations_dir = Path(os.environ.get("MIGRATIONS_DIR", "/migrations"))
    container = PostgresContainer("postgres:17-alpine", driver="psycopg2")
    container.start()
    try:
        engine = create_engine(container.get_connection_url(), pool_pre_ping=True)
        with engine.begin() as conn:
            for path in sorted(migrations_dir.glob("V*.sql")):
                conn.execute(text(path.read_text()))
        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()
        try:
            yield session
        finally:
            session.close()
            engine.dispose()
    finally:
        container.stop()


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_plan_with_real_ollama(
    real_db_session: Session, ollama_host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifie que generate_plan fonctionne contre un vrai Ollama (1 generation)."""
    monkeypatch.setattr(settings, "ollama_host", ollama_host)
    inputs = PlanInputs(
        user_id=999,
        objective="balance",
        duration_days=1,
        diet_type="omnivore",
    )

    plan = await generate_plan(inputs, real_db_session)

    assert plan.fallback is False
    assert len(plan.days) >= 1
    assert plan.days[0].meals[0].macros.calories > 0


@pytest.mark.slow
@pytest.mark.asyncio
async def test_generate_recommendation_with_real_ollama(
    real_db_session: Session, ollama_host: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(settings, "ollama_host", ollama_host)
    tags = [
        ImbalanceTag(
            nutrient=Nutrient.protein_g,
            status=ImbalanceStatus.deficit,
            delta_pct=-0.40,
            target_value=50.0,
            actual_value=30.0,
            unit="g",
        )
    ]

    text_reco = await generate_recommendation(
        ctx_list=tags,
        health_goal=HealthGoal.muscle_gain,
        db=real_db_session,
    )

    assert isinstance(text_reco, str)
    assert len(text_reco) > 0
