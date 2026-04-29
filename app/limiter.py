from __future__ import annotations

from fastapi import HTTPException, Request
from slowapi import Limiter

from app.services import jwt_decoder


def _user_key(request: Request) -> str:
    """Clef de rate limit : user_id du JWT, sinon IP en fallback.

    Si le JWT est absent ou invalide, l'endpoint repondra 401 ; le rate limit
    par IP empeche juste un attaquant non authentifie d'epuiser le service.
    """
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            identity = jwt_decoder.decode(auth.removeprefix("Bearer "))
        except HTTPException:
            pass
        else:
            return f"user:{identity.user_id}"
    if request.client is not None:
        return f"ip:{request.client.host}"
    return "anonymous"


limiter = Limiter(key_func=_user_key)
