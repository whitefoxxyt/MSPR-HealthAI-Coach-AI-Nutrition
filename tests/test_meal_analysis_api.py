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
    """Pizza forte en calories et glucides pour declencher des imbalances."""
    db.execute(
        text(
            "INSERT INTO nutrition_entries "
            "(food_name, calories, protein_g, carbs_g, fat_g, fiber_g, source) "
            "VALUES ('pizza', 1300, 30, 160, 50, 5, 'TEST')"
        )
    )
    db.commit()


def _seed_full_profile(
    db: Session,
    user_id: int,
    *,
    health_goal: str | None = "balance",
    gender: str = "male",
    age: int = 30,
    weight_kg: float = 80.0,
    height_cm: float = 180.0,
    activity_level: str = "moderate",
) -> None:
    """Profil complet avec biometrie : TDEE calculable -> imbalances detectables."""
    db.execute(
        text(
            "INSERT INTO nutrition_goals "
            "(user_id, health_goal, gender, age, weight_kg, height_cm, activity_level) "
            "VALUES (:uid, :goal, :g, :a, :w, :h, :al)"
        ),
        {
            "uid": user_id,
            "goal": health_goal,
            "g": gender,
            "a": age,
            "w": weight_kg,
            "h": height_cm,
            "al": activity_level,
        },
    )
    db.commit()


def _seed_partial_profile(db: Session, user_id: int) -> None:
    """Profil sans biometrie : detect() doit retourner [] (profile_completion_required)."""
    db.execute(
        text("INSERT INTO nutrition_goals (user_id, health_goal) VALUES (:uid, :goal)"),
        {"uid": user_id, "goal": "balance"},
    )
    db.commit()


def test_analyze_meal_nominal_returns_imbalance_tags_and_single_recommendation(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 42
    _seed_pizza(db_session)
    _seed_full_profile(db_session, user_id)

    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200,
        json={
            "response": "Reduis la portion et ajoute des legumes verts.",
            "done": True,
        },
    )

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()

    # Profil complet : pas de demande de completion.
    assert body["profile_completion_required"] is False
    assert body["missing_fields"] == []
    assert body["fallback"] is False

    # Pizza riche en calories et glucides : detect() emet au moins 1 tag.
    assert len(body["imbalances"]) >= 1
    assert all(
        {
            "nutrient",
            "status",
            "delta_pct",
            "target_value",
            "actual_value",
            "unit",
        }.issubset(t)
        for t in body["imbalances"]
    )

    # Phrase deterministe par tag.
    assert len(body["imbalances_text"]) == len(body["imbalances"])

    # Synthese unique par le LLM (1 appel, 1 phrase).
    assert route.call_count == 1
    assert body["recommendations"] == ["Reduis la portion et ajoute des legumes verts."]


def test_analyze_meal_persists_imbalances_jsonb(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 99
    _seed_pizza(db_session)
    _seed_full_profile(db_session, user_id)

    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json={"response": "Conseil unique.", "done": True}
    )

    client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    row = db_session.execute(
        text("SELECT imbalances FROM meal_analyses WHERE user_id = :uid"),
        {"uid": user_id},
    ).fetchone()
    assert row is not None
    persisted = row.imbalances
    assert isinstance(persisted, list) and len(persisted) >= 1
    first = persisted[0]
    assert {"nutrient", "status", "delta_pct"}.issubset(first)


def test_analyze_meal_returns_completion_required_when_profile_incomplete(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 50
    _seed_pizza(db_session)
    _seed_partial_profile(db_session, user_id)  # pas de biometrie

    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json={"response": "ne devrait pas etre appele.", "done": True}
    )

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile_completion_required"] is True
    # Champs biometriques manquants -> liste populee.
    assert set(body["missing_fields"]) >= {
        "gender",
        "age",
        "weight_kg",
        "height_cm",
        "activity_level",
    }
    assert body["imbalances"] == []
    assert body["imbalances_text"] == []
    assert body["recommendations"] == []
    # Aucun appel LLM tant que le profil n'est pas complet.
    assert route.call_count == 0


def test_analyze_meal_returns_completion_required_when_no_profile(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 51
    _seed_pizza(db_session)
    # Pas de profil du tout : aucune ligne en BDD.

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile_completion_required"] is True
    assert body["recommendations"] == []
    assert body["imbalances"] == []


def test_analyze_meal_falls_back_to_matrix_when_llm_down(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 43
    _seed_pizza(db_session)
    _seed_full_profile(db_session, user_id, health_goal="weight_loss")

    # Tous les essais Ollama echouent : llm_client epuise ses retries puis fallback matrice.
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        503, json={"error": "down"}
    )

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is True
    # Une recommandation synthetique unique (matrice concatenee), pas une par imbalance.
    assert len(body["recommendations"]) == 1
    # Hash NULL en mode fallback : un appel ulterieur retentera le LLM.
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
    _seed_full_profile(db_session, user_id, health_goal="balance")

    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json={"response": "Garde une portion raisonnable.", "done": True}
    )

    headers = {"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"}

    def _files() -> dict[str, tuple[str, bytes, str]]:
        return {"photo": ("pizza.png", _png_bytes(), "image/png")}

    first = client.post("/api/v1/analyze-meal", files=_files(), headers=headers)
    assert first.status_code == 200
    assert first.json()["fallback"] is False
    assert route.call_count == 1  # 1 seul appel pour le synthetique

    second = client.post("/api/v1/analyze-meal", files=_files(), headers=headers)
    assert second.status_code == 200
    body2 = second.json()
    assert body2["fallback"] is False
    assert body2["recommendations"] == first.json()["recommendations"]
    # Cache hit : aucun appel LLM supplementaire.
    assert route.call_count == 1


def test_analyze_meal_uses_balance_when_health_goal_missing(
    client: TestClient,
    db_session: Session,
    mock_classifier: Callable[..., list[dict[str, Any]]],
    mock_ollama: respx.MockRouter,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 45
    _seed_pizza(db_session)
    # Profil avec biometrie complete mais health_goal NULL.
    _seed_full_profile(db_session, user_id, health_goal=None)

    captured_prompts: list[str] = []

    def _record(request: httpx.Request) -> httpx.Response:
        captured_prompts.append(httpx.Request.read(request).decode())
        return httpx.Response(
            200, json={"response": "Conseil generique.", "done": True}
        )

    mock_ollama.post(re.compile(r".*/api/generate$")).mock(side_effect=_record)

    response = client.post(
        "/api/v1/analyze-meal",
        files={"photo": ("pizza.png", _png_bytes(), "image/png")},
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is False
    assert body["recommendations"] == ["Conseil generique."]
    # Le LLM a bien ete appele avec health_goal=balance.
    assert captured_prompts, "Aucun appel LLM enregistre"
    assert all("balance" in p for p in captured_prompts)
