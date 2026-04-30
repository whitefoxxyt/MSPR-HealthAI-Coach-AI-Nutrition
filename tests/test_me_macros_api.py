from __future__ import annotations

from collections.abc import Callable, Generator

import pytest
from fastapi.testclient import TestClient
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


def _full_profile_payload() -> dict:
    return {
        "health_goal": "balance",
        "gender": "male",
        "age": 30,
        "weight_kg": 80,
        "height_cm": 180,
        "activity_level": "moderate",
    }


def test_get_macros_with_complete_profile_returns_tdee_and_macros(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=42)
    auth = {"Authorization": f"Bearer {token}"}
    client.put(
        "/api/v1/nutrition-goals/me",
        json=_full_profile_payload(),
        headers=auth,
    )

    response = client.get("/api/v1/me/macros", headers=auth)

    assert response.status_code == 200
    body = response.json()
    # Mifflin-St Jeor homme 30 ans, 80 kg, 180 cm -> BMR 1780 ; moderate -> TDEE 2759.
    # Repartition balance : 50 % glucides, 20 % proteines, 30 % lipides.
    assert body["profile_completion_required"] is False
    assert body["missing_fields"] == []
    assert body["tdee"] == pytest.approx(2759.0)
    macros = body["macros"]
    assert macros["calories"] == pytest.approx(2759.0)
    assert macros["protein_g"] == pytest.approx(2759.0 * 0.20 / 4.0)
    assert macros["carbs_g"] == pytest.approx(2759.0 * 0.50 / 4.0)
    assert macros["fat_g"] == pytest.approx(2759.0 * 0.30 / 9.0)


def test_get_macros_without_profile_returns_completion_required(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=99)

    response = client.get(
        "/api/v1/me/macros",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["profile_completion_required"] is True
    assert set(body["missing_fields"]) == {
        "gender",
        "age",
        "weight_kg",
        "height_cm",
        "activity_level",
    }
    assert body["tdee"] is None
    assert body["macros"] is None


def test_get_macros_with_partial_profile_lists_only_missing_fields(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=11)
    auth = {"Authorization": f"Bearer {token}"}
    # Profil avec health_goal + age + height seulement.
    client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "weight_loss", "age": 28, "height_cm": 170},
        headers=auth,
    )

    response = client.get("/api/v1/me/macros", headers=auth)

    assert response.status_code == 200
    body = response.json()
    assert body["profile_completion_required"] is True
    assert set(body["missing_fields"]) == {"gender", "weight_kg", "activity_level"}
    assert body["tdee"] is None


def test_get_macros_without_authorization_returns_401(client: TestClient) -> None:
    response = client.get("/api/v1/me/macros")
    assert response.status_code == 401


def test_get_macros_with_invalid_jwt_returns_401(client: TestClient) -> None:
    response = client.get(
        "/api/v1/me/macros",
        headers={"Authorization": "Bearer not-a-real-jwt"},
    )
    assert response.status_code == 401


def test_get_macros_isolates_users(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    alice = valid_jwt(user_id=100)
    bob = valid_jwt(user_id=200)
    # Alice configure un profil complet ; Bob non.
    client.put(
        "/api/v1/nutrition-goals/me",
        json=_full_profile_payload(),
        headers={"Authorization": f"Bearer {alice}"},
    )

    bob_response = client.get(
        "/api/v1/me/macros",
        headers={"Authorization": f"Bearer {bob}"},
    )

    assert bob_response.status_code == 200
    assert bob_response.json()["profile_completion_required"] is True
