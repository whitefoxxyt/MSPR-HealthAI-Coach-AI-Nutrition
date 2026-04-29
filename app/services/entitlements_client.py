from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import httpx
from cachetools import TTLCache

from app.config import settings

logger = logging.getLogger(__name__)

Tier = Literal["free", "premium", "premium_plus"]

_TIMEOUT_S = 3.0
_CACHE_TTL_S = 60.0
_CACHE_MAXSIZE = 10000


@dataclass
class Entitlements:
    tier: Tier
    expires_at: datetime | None
    features: list[str]


# Cache TTL in-memory : 60s par user_id, plafonne pour eviter une fuite memoire.
_cache: TTLCache[str, Entitlements] = TTLCache(maxsize=_CACHE_MAXSIZE, ttl=_CACHE_TTL_S)
# Dernier resultat connu par user_id, conserve meme apres expiration du cache TTL,
# pour servir de fallback degrade quand MSPR-AUTH timeout ou est injoignable.
_stale: dict[str, Entitlements] = {}

_FREE = Entitlements(tier="free", expires_at=None, features=[])
_VALID_TIERS = {"free", "premium", "premium_plus"}


def _parse(payload: dict) -> Entitlements:
    tier = payload.get("tier")
    if tier not in _VALID_TIERS:
        raise ValueError(f"tier invalide: {tier!r}")
    raw_exp = payload.get("expires_at")
    expires_at = datetime.fromisoformat(raw_exp) if raw_exp else None
    features = payload.get("features") or []
    if not isinstance(features, list):
        raise ValueError("features doit etre une liste")
    return Entitlements(tier=tier, expires_at=expires_at, features=list(features))


async def get_entitlements(user_id: str, jwt: str) -> Entitlements:
    cached = _cache.get(user_id)
    if cached is not None:
        return cached

    url = f"{settings.auth_api_url.rstrip('/')}/api/entitlements/me"
    headers = {"Authorization": f"Bearer {jwt}"}

    try:
        async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers)
    except httpx.RequestError as exc:
        logger.warning(
            "entitlements: erreur reseau MSPR-AUTH (%s), degrade pour user_id=%s",
            exc,
            user_id,
        )
        return _stale.get(user_id, _FREE)

    if response.status_code >= 400:
        logger.warning(
            "entitlements: status %d de MSPR-AUTH, degrade vers free pour user_id=%s",
            response.status_code,
            user_id,
        )
        return _FREE

    try:
        ent = _parse(response.json())
    except (ValueError, TypeError) as exc:
        logger.warning(
            "entitlements: reponse invalide de MSPR-AUTH (%s), degrade vers free pour user_id=%s",
            exc,
            user_id,
        )
        return _FREE

    _cache[user_id] = ent
    _stale[user_id] = ent
    return ent
