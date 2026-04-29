from __future__ import annotations

from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import HTTPException

from app.config import settings
from app.services.jwt_decoder import UserIdentity, decode

SECRET = "test-secret-please-change-in-prod-32+chars"


def _encode(payload: dict, secret: str = SECRET, algorithm: str = "HS256") -> str:
    return jwt.encode(payload, secret, algorithm=algorithm)


def _future_exp(seconds: int = 3600) -> int:
    return int((datetime.now(tz=timezone.utc) + timedelta(seconds=seconds)).timestamp())


@pytest.fixture(autouse=True)
def _set_secret(monkeypatch):
    monkeypatch.setattr(settings, "better_auth_secret", SECRET)


def test_decode_valid_token_returns_identity():
    token = _encode({"sub": "user-123", "email": "alice@example.com", "exp": _future_exp()})

    identity = decode(token)

    assert isinstance(identity, UserIdentity)
    assert identity.user_id == "user-123"
    assert identity.email == "alice@example.com"


def test_decode_invalid_signature_raises_401():
    token = _encode(
        {"sub": "user-123", "exp": _future_exp()},
        secret="wrong-secret-not-the-server-one-32+chars",
    )

    with pytest.raises(HTTPException) as exc:
        decode(token)

    assert exc.value.status_code == 401


def test_decode_expired_token_raises_401():
    past_exp = int((datetime.now(tz=timezone.utc) - timedelta(seconds=60)).timestamp())
    token = _encode({"sub": "user-123", "exp": past_exp})

    with pytest.raises(HTTPException) as exc:
        decode(token)

    assert exc.value.status_code == 401


def test_decode_missing_sub_claim_raises_401():
    token = _encode({"email": "alice@example.com", "exp": _future_exp()})

    with pytest.raises(HTTPException) as exc:
        decode(token)

    assert exc.value.status_code == 401


def test_decode_alg_none_is_rejected():
    # token forge avec alg=none, attaque classique
    token = jwt.encode(
        {"sub": "attacker", "exp": _future_exp()},
        key="",
        algorithm="none",
    )

    with pytest.raises(HTTPException) as exc:
        decode(token)

    assert exc.value.status_code == 401


def test_decode_unexpected_hs_algorithm_is_rejected():
    # token signe en HS512 alors que le decodeur exige HS256
    token = jwt.encode(
        {"sub": "user-123", "exp": _future_exp()},
        SECRET,
        algorithm="HS512",
    )

    with pytest.raises(HTTPException) as exc:
        decode(token)

    assert exc.value.status_code == 401


def test_decode_missing_exp_claim_raises_401():
    token = _encode({"sub": "user-123"})

    with pytest.raises(HTTPException) as exc:
        decode(token)

    assert exc.value.status_code == 401


def test_decode_with_empty_secret_fails_closed(monkeypatch):
    # secret vide = misconfiguration ; doit refuser plutot que de valider n'importe quoi
    monkeypatch.setattr(settings, "better_auth_secret", "")
    token = jwt.encode({"sub": "user-123", "exp": _future_exp()}, "", algorithm="HS256")

    with pytest.raises(HTTPException) as exc:
        decode(token)

    assert exc.value.status_code == 500
