from __future__ import annotations

from collections.abc import Callable, Generator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.main import app
from tests.conftest import TEST_AUTH_SECRET


@pytest.fixture(autouse=True)
def _set_secret_and_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "better_auth_secret", TEST_AUTH_SECRET)
    monkeypatch.setattr(settings, "default_llm", "mistral")


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_get_preferences_without_profile_returns_env_default(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id="42")

    response = client.get(
        "/api/v1/me/preferences",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["preferred_llm"] is None
    assert body["effective_llm"] == "mistral"


def test_patch_preferences_sets_user_pref_and_get_returns_it(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id="43")
    auth = {"Authorization": f"Bearer {token}"}

    patch = client.patch(
        "/api/v1/me/preferences",
        json={"preferred_llm": "ollama"},
        headers=auth,
    )

    assert patch.status_code == 200, patch.text
    assert patch.json() == {"preferred_llm": "ollama", "effective_llm": "ollama"}

    get = client.get("/api/v1/me/preferences", headers=auth)
    assert get.status_code == 200
    assert get.json() == {"preferred_llm": "ollama", "effective_llm": "ollama"}


def test_patch_preferences_with_unknown_backend_returns_422(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    token = valid_jwt(user_id="44")

    response = client.patch(
        "/api/v1/me/preferences",
        json={"preferred_llm": "anthropic"},
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 422


def test_patch_preferences_with_null_resets_to_default(
    client: TestClient,
    valid_jwt: Callable[..., str],
    db_session: Session,
) -> None:
    token = valid_jwt(user_id="45")
    auth = {"Authorization": f"Bearer {token}"}
    client.patch(
        "/api/v1/me/preferences", json={"preferred_llm": "ollama"}, headers=auth
    )

    reset = client.patch(
        "/api/v1/me/preferences", json={"preferred_llm": None}, headers=auth
    )

    assert reset.status_code == 200
    assert reset.json() == {"preferred_llm": None, "effective_llm": "mistral"}
    row = db_session.execute(
        text("SELECT preferred_llm FROM nutrition_goals WHERE user_id = '45'")
    ).fetchone()
    assert row.preferred_llm is None


def test_get_preferences_without_authorization_returns_401(
    client: TestClient,
) -> None:
    assert client.get("/api/v1/me/preferences").status_code == 401


def test_patch_preferences_without_authorization_returns_401(
    client: TestClient,
) -> None:
    response = client.patch("/api/v1/me/preferences", json={"preferred_llm": "ollama"})
    assert response.status_code == 401


def test_preferences_isolated_per_user(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    alice = valid_jwt(user_id="100")
    bob = valid_jwt(user_id="200")
    client.patch(
        "/api/v1/me/preferences",
        json={"preferred_llm": "ollama"},
        headers={"Authorization": f"Bearer {alice}"},
    )

    bob_view = client.get(
        "/api/v1/me/preferences",
        headers={"Authorization": f"Bearer {bob}"},
    )

    assert bob_view.status_code == 200
    assert bob_view.json()["preferred_llm"] is None
