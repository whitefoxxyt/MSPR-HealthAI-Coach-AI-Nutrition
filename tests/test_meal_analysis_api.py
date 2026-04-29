from __future__ import annotations

import re
from collections.abc import Callable, Generator
from io import BytesIO
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.main import app
from tests.conftest import TEST_AUTH_SECRET


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "better_auth_secret", TEST_AUTH_SECRET)


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def _png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGB", (4, 4), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _seed_pizza(db: Session) -> None:
    """Pizza forte en calories et glucides (declenche calories_high + carbs_high
    quand l'utilisateur a calories_target=2000 et carbs_g=200)."""
    db.execute(
        text(
            "INSERT INTO nutrition_entries "
            "(food_name, calories, protein_g, carbs_g, fat_g, fiber_g, source) "
            "VALUES ('pizza', 1300, 30, 160, 50, 5, 'TEST')"
        )
    )
    db.commit()


def _seed_profile(
    db: Session,
    user_id: int,
    *,
    health_goal: str | None = "muscle_gain",
    calories_target: int = 2000,
    protein_g: float = 100.0,
    carbs_g: float = 200.0,
    fat_g: float = 80.0,
) -> None:
    db.execute(
        text(
            "INSERT INTO nutrition_goals "
            "(user_id, health_goal, calories_target, protein_g, carbs_g, fat_g) "
            "VALUES (:uid, :goal, :cal, :prot, :carb, :fat)"
        ),
        {
            "uid": user_id,
            "goal": health_goal,
            "cal": calories_target,
            "prot": protein_g,
            "carb": carbs_g,
            "fat": fat_g,
        },
    )
    db.commit()


def test_analyze_meal_nominal_returns_llm_recommendations(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 42
    _seed_pizza(db_session)
    _seed_profile(db_session, user_id)

    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json={"response": "Reduis la portion au prochain repas.", "done": True}
    )

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is False
    # Pizza : calories 1300/2000 = 65 % > 60 % et glucides 160/200 = 80 % > 70 %.
    # Deux desequilibres -> deux appels LLM -> deux recommandations.
    assert len(body["recommendations"]) == 2
    assert all(r == "Reduis la portion au prochain repas." for r in body["recommendations"])
    assert route.call_count == 2

    row = db_session.execute(
        text(
            "SELECT user_id, recommendations, recommendations_hash "
            "FROM meal_analyses WHERE user_id = :uid"
        ),
        {"uid": user_id},
    ).fetchone()
    assert row is not None
    assert row.user_id == user_id
    assert len(row.recommendations) == 2
    assert row.recommendations_hash is not None
    assert len(row.recommendations_hash) == 64


def test_analyze_meal_falls_back_to_matrix_when_llm_down(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 43
    _seed_pizza(db_session)
    _seed_profile(db_session, user_id, health_goal="weight_loss")

    # 503 sur tous les essais : llm_client epuise ses retries puis bascule sur le fallback.
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(503, json={"error": "down"})

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is True
    assert len(body["recommendations"]) == 2
    # Les phrases de la matrice contiennent des marqueurs (objectif perte de poids).
    assert any("perte de poids" in r.lower() for r in body["recommendations"])

    # Mode fallback : on ne pose pas le hash (un futur appel doit retenter le LLM).
    row = db_session.execute(
        text("SELECT recommendations_hash FROM meal_analyses WHERE user_id = :uid"),
        {"uid": user_id},
    ).fetchone()
    assert row is not None
    assert row.recommendations_hash is None


def test_analyze_meal_cache_hit_skips_llm_on_second_call(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 44
    _seed_pizza(db_session)
    _seed_profile(db_session, user_id, health_goal="balance")

    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json={"response": "Garde une portion raisonnable.", "done": True}
    )

    headers = {"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"}

    def _files() -> dict[str, tuple[str, bytes, str]]:
        return {"photo": ("pizza.png", _png_bytes(), "image/png")}

    first = client.post("/api/v1/analyze-meal", files=_files(), headers=headers)
    assert first.status_code == 200
    assert first.json()["fallback"] is False
    first_calls = route.call_count
    assert first_calls == 2

    second = client.post("/api/v1/analyze-meal", files=_files(), headers=headers)
    assert second.status_code == 200
    body2 = second.json()
    assert body2["fallback"] is False
    assert body2["recommendations"] == first.json()["recommendations"]
    # Cache hit : aucun appel LLM supplementaire.
    assert route.call_count == first_calls


def test_analyze_meal_uses_balance_when_health_goal_missing(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 45
    _seed_pizza(db_session)
    # Profil avec macros mais sans health_goal (NULL en BDD).
    _seed_profile(db_session, user_id, health_goal=None)

    captured_prompts: list[str] = []

    def _record(request: httpx.Request) -> httpx.Response:
        captured_prompts.append(httpx.Request.read(request).decode())
        return httpx.Response(200, json={"response": "Conseil generique.", "done": True})

    mock_ollama.post(re.compile(r".*/api/generate$")).mock(side_effect=_record)

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is False
    assert len(body["recommendations"]) >= 1
    # Le LLM a bien ete appele avec health_goal=balance.
    assert captured_prompts, "Aucun appel LLM enregistre"
    assert all("balance" in p for p in captured_prompts)
