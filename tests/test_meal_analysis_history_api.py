from __future__ import annotations

import json
from collections.abc import Callable, Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.main import app
from tests.conftest import TEST_AUTH_SECRET


def _seed_analysis(
    db: Session,
    user_id: int,
    *,
    foods: list | None = None,
    macros: dict | None = None,
    recommendations: list[str] | None = None,
) -> None:
    db.execute(
        text(
            "INSERT INTO meal_analyses "
            "(user_id, detected_foods, macros, recommendations) "
            "VALUES (:uid, CAST(:foods AS JSONB), CAST(:macros AS JSONB), CAST(:recos AS JSONB))"
        ),
        {
            "uid": user_id,
            "foods": json.dumps(foods or []),
            "macros": json.dumps(macros or {}),
            "recos": json.dumps(recommendations or []),
        },
    )
    db.commit()


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


def test_returns_empty_when_user_has_no_analyses(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    response = client.get(
        "/api/v1/meal-analyses/me",
        headers={"Authorization": f"Bearer {valid_jwt(user_id=42)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {"items": [], "total": 0, "limit": 20, "offset": 0}


def test_returns_only_analyses_of_authenticated_user(
    client: TestClient,
    db_session: Session,
    valid_jwt: Callable[..., str],
) -> None:
    user_a, user_b = 100, 200
    for _ in range(5):
        _seed_analysis(db_session, user_a)
    for _ in range(3):
        _seed_analysis(db_session, user_b)

    response = client.get(
        "/api/v1/meal-analyses/me",
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_a)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 5
    assert len(body["items"]) == 5


def test_items_sorted_by_created_at_desc(
    client: TestClient,
    db_session: Session,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 7
    db_session.execute(
        text(
            "INSERT INTO meal_analyses (user_id, created_at) VALUES "
            "(:uid, '2025-01-01 10:00:00'), "
            "(:uid, '2025-03-01 10:00:00'), "
            "(:uid, '2025-02-01 10:00:00')"
        ),
        {"uid": user_id},
    )
    db_session.commit()

    response = client.get(
        "/api/v1/meal-analyses/me",
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200
    timestamps = [item["created_at"] for item in response.json()["items"]]
    assert timestamps == sorted(timestamps, reverse=True)


def test_pagination_returns_window_and_total(
    client: TestClient,
    db_session: Session,
    valid_jwt: Callable[..., str],
) -> None:
    user_id = 8
    # 5 analyses datees du plus ancien au plus recent.
    db_session.execute(
        text(
            "INSERT INTO meal_analyses (user_id, created_at) VALUES "
            "(:uid, '2025-01-01'), (:uid, '2025-01-02'), (:uid, '2025-01-03'), "
            "(:uid, '2025-01-04'), (:uid, '2025-01-05')"
        ),
        {"uid": user_id},
    )
    db_session.commit()

    response = client.get(
        "/api/v1/meal-analyses/me?limit=2&offset=1",
        headers={"Authorization": f"Bearer {valid_jwt(user_id=user_id)}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 5
    assert body["limit"] == 2
    assert body["offset"] == 1
    # Tri DESC : items[0]=2025-01-05 (offset 0), items[1]=2025-01-04 (offset 1).
    # offset=1 saute le plus recent, limit=2 prend les deux suivants.
    timestamps = [item["created_at"] for item in body["items"]]
    assert len(timestamps) == 2
    assert timestamps[0].startswith("2025-01-04")
    assert timestamps[1].startswith("2025-01-03")


@pytest.mark.parametrize(
    "query",
    ["limit=0", "limit=-1", "limit=101", "offset=-1"],
)
def test_invalid_pagination_params_return_422(
    client: TestClient,
    valid_jwt: Callable[..., str],
    query: str,
) -> None:
    response = client.get(
        f"/api/v1/meal-analyses/me?{query}",
        headers={"Authorization": f"Bearer {valid_jwt(user_id=1)}"},
    )
    assert response.status_code == 422, response.text


def test_missing_authorization_header_returns_401(client: TestClient) -> None:
    response = client.get("/api/v1/meal-analyses/me")
    assert response.status_code == 401


def test_invalid_jwt_returns_401(client: TestClient) -> None:
    response = client.get(
        "/api/v1/meal-analyses/me",
        headers={"Authorization": "Bearer not-a-real-jwt"},
    )
    assert response.status_code == 401
