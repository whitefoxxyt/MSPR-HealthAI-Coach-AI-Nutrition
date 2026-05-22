from __future__ import annotations

from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from app.config import settings
from app.limiter import _user_key
from tests.conftest import TEST_AUTH_SECRET


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "better_auth_secret", TEST_AUTH_SECRET)


def _request(headers: dict[str, str] | None = None, client_host: str | None = "1.2.3.4") -> MagicMock:
    req = MagicMock()
    req.headers.get = (headers or {}).get
    if client_host is None:
        req.client = None
    else:
        req.client = MagicMock(host=client_host)
    return req


def test_key_uses_user_id_for_valid_jwt(valid_jwt: Callable[..., str]) -> None:
    token = valid_jwt(user_id="42")
    key = _user_key(_request({"Authorization": f"Bearer {token}"}))
    assert key == "user:42"


def test_key_falls_back_to_ip_when_no_authorization() -> None:
    key = _user_key(_request({}))
    assert key == "ip:1.2.3.4"


def test_key_falls_back_to_ip_when_jwt_invalid() -> None:
    key = _user_key(_request({"Authorization": "Bearer not-a-jwt"}))
    assert key == "ip:1.2.3.4"


def test_key_falls_back_to_anonymous_without_client() -> None:
    key = _user_key(_request({}, client_host=None))
    assert key == "anonymous"
