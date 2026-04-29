from __future__ import annotations

from dataclasses import dataclass

import jwt
from fastapi import HTTPException

from app.config import settings


@dataclass
class UserIdentity:
    user_id: str
    email: str | None


def decode(token: str) -> UserIdentity:
    # Fail-closed : un secret vide rendrait n'importe quel JWT signe avec "" valide.
    if not settings.better_auth_secret:
        raise HTTPException(status_code=500, detail="JWT secret non configure.")
    try:
        payload = jwt.decode(
            token,
            settings.better_auth_secret,
            algorithms=["HS256"],
            options={"require": ["sub", "exp"]},
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Token invalide.") from exc
    return UserIdentity(user_id=payload["sub"], email=payload.get("email"))
