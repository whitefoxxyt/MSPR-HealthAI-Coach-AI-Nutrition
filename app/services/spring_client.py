from __future__ import annotations

import httpx

from app.config import settings


async def get_user_me(jwt_token: str) -> dict:
    """
    Appelle GET /users/me sur l'API Spring Boot avec le JWT fourni.

    Retourne le profil utilisateur (id, objectifs nutritionnels, etc.).
    Lève une httpx.HTTPStatusError si la réponse n'est pas 2xx.
    """
    url = f"{settings.spring_api_url}/api/users/me"
    headers = {"Authorization": f"Bearer {jwt_token}"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
