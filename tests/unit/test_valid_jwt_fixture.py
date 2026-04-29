from __future__ import annotations

from collections.abc import Callable

from jose import jwt

from tests.conftest import TEST_AUTH_SECRET


def test_valid_jwt_default_claims(valid_jwt: Callable[..., str]) -> None:
    token = valid_jwt()

    payload = jwt.decode(token, TEST_AUTH_SECRET, algorithms=["HS256"])
    assert payload["sub"] == "1"
    assert payload["email"] == "test@example.com"
    assert "exp" in payload and "iat" in payload


def test_valid_jwt_custom_user_id(valid_jwt: Callable[..., str]) -> None:
    token = valid_jwt(user_id=42, email=None, extra_claims={"role": "premium"})

    payload = jwt.decode(token, TEST_AUTH_SECRET, algorithms=["HS256"])
    assert payload["sub"] == "42"
    assert "email" not in payload
    assert payload["role"] == "premium"
