from __future__ import annotations

from collections.abc import Callable, Generator
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from jose import jwt
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.main import app
from tests.conftest import TEST_AUTH_SECRET


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    """Aligne le secret JWT du service sur celui de la fixture valid_jwt."""
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


def test_put_creates_profile_and_returns_data(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=42)
    payload = {
        "health_goal": "muscle_gain",
        "calories_target": 2400,
        "protein_g": 180,
        "carbs_g": 280,
        "fat_g": 70,
        "allergies": ["arachides"],
        "diet_type": "omnivore",
    }

    response = client.put(
        "/api/v1/nutrition-goals/me",
        json=payload,
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 42
    assert body["health_goal"] == "muscle_gain"
    assert body["calories_target"] == 2400
    assert body["allergies"] == ["arachides"]
    assert body["diet_type"] == "omnivore"


def test_get_returns_404_when_no_profile(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=42)

    response = client.get(
        "/api/v1/nutrition-goals/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 404


def test_get_returns_profile_after_put(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=7)
    auth = {"Authorization": f"Bearer {token}"}
    client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "weight_loss", "calories_target": 1800, "diet_type": "vegetarien"},
        headers=auth,
    )

    response = client.get("/api/v1/nutrition-goals/me", headers=auth)

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 7
    assert body["health_goal"] == "weight_loss"
    assert body["calories_target"] == 1800
    assert body["diet_type"] == "vegetarien"


def test_second_put_updates_profile(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=12)
    auth = {"Authorization": f"Bearer {token}"}
    client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "balance", "calories_target": 2000},
        headers=auth,
    )

    response = client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "sport_performance", "calories_target": 2800},
        headers=auth,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 12
    assert body["health_goal"] == "sport_performance"
    assert body["calories_target"] == 2800

    get_response = client.get("/api/v1/nutrition-goals/me", headers=auth)
    assert get_response.json()["health_goal"] == "sport_performance"


def test_put_with_invalid_health_goal_returns_422(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=42)

    response = client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "marathon"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


def test_user_cannot_read_another_users_profile(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    alice_token = valid_jwt(user_id=100)
    bob_token = valid_jwt(user_id=200)

    client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "muscle_gain", "calories_target": 3000},
        headers={"Authorization": f"Bearer {alice_token}"},
    )

    bob_get = client.get(
        "/api/v1/nutrition-goals/me",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert bob_get.status_code == 404


def test_user_cannot_overwrite_another_users_profile(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    alice_token = valid_jwt(user_id=100)
    bob_token = valid_jwt(user_id=200)

    client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "muscle_gain", "calories_target": 3000},
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    bob_put = client.put(
        "/api/v1/nutrition-goals/me",
        json={"health_goal": "weight_loss", "calories_target": 1500},
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert bob_put.status_code == 200
    assert bob_put.json()["user_id"] == 200

    alice_get = client.get(
        "/api/v1/nutrition-goals/me",
        headers={"Authorization": f"Bearer {alice_token}"},
    )
    assert alice_get.status_code == 200
    assert alice_get.json()["health_goal"] == "muscle_gain"
    assert alice_get.json()["calories_target"] == 3000


def test_missing_authorization_header_returns_401(client: TestClient) -> None:
    get_resp = client.get("/api/v1/nutrition-goals/me")
    put_resp = client.put("/api/v1/nutrition-goals/me", json={})
    assert get_resp.status_code == 401
    assert put_resp.status_code == 401


def test_invalid_jwt_returns_401(client: TestClient) -> None:
    response = client.get(
        "/api/v1/nutrition-goals/me",
        headers={"Authorization": "Bearer not-a-real-jwt"},
    )
    assert response.status_code == 401


def test_expired_jwt_returns_401(client: TestClient) -> None:
    past = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    expired = jwt.encode(
        {"sub": "1", "iat": int(past.timestamp()), "exp": int(past.timestamp())},
        TEST_AUTH_SECRET,
        algorithm="HS256",
    )
    response = client.get(
        "/api/v1/nutrition-goals/me",
        headers={"Authorization": f"Bearer {expired}"},
    )
    assert response.status_code == 401


def test_jwt_with_non_numeric_sub_returns_401(client: TestClient) -> None:
    now = datetime.now(tz=timezone.utc)
    token = jwt.encode(
        {
            "sub": "uuid-not-an-int",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(hours=1)).timestamp()),
        },
        TEST_AUTH_SECRET,
        algorithm="HS256",
    )
    response = client.get(
        "/api/v1/nutrition-goals/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 401


def test_put_accepts_all_health_goal_values(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    for idx, goal in enumerate(
        ["weight_loss", "muscle_gain", "balance", "sport_performance"]
    ):
        token = valid_jwt(user_id=500 + idx)
        response = client.put(
            "/api/v1/nutrition-goals/me",
            json={"health_goal": goal},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["health_goal"] == goal


def test_put_with_minimal_payload_succeeds(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id=42)

    response = client.put(
        "/api/v1/nutrition-goals/me",
        json={},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == 42
    assert body["health_goal"] is None
    assert body["allergies"] == []
