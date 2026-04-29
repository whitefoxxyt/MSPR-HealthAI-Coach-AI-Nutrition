from datetime import datetime, timezone

import httpx
from fastapi import APIRouter

from app.config import settings
from app.db.session import check_postgres

router = APIRouter()


async def check_ollama() -> str:
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            resp = await client.get(f"{settings.ollama_host}/api/tags")
            return "up" if resp.status_code == 200 else "down"
    except Exception:
        return "down"


_HEALTH_DESCRIPTION = """
Verifie l'etat operationnel du service et de ses dependances critiques.

Renvoie un statut global, l'etat individuel de PostgreSQL et d'Ollama, ainsi qu'un horodatage UTC.

- `status` : `ok` si toutes les dependances sont joignables, `degraded` sinon.
- `postgres` : `up` ou `down` (timeout 3 s sur SELECT 1).
- `ollama` : `up` ou `down` (timeout 3 s sur GET /api/tags).

Cet endpoint est non authentifie et ne consomme aucune ressource lourde. Il est utilise par les sondes de liveness/readiness Docker et par le monitoring.
"""

_HEALTH_RESPONSE_EXAMPLE = {
    "status": "ok",
    "postgres": "up",
    "ollama": "up",
    "timestamp": "2026-04-29T10:15:00.123456+00:00",
}


@router.get(
    "/health",
    tags=["Sante"],
    summary="Healthcheck du service et de ses dependances",
    description=_HEALTH_DESCRIPTION,
    responses={
        200: {
            "description": "Statut operationnel (ok ou degraded selon les dependances).",
            "content": {"application/json": {"example": _HEALTH_RESPONSE_EXAMPLE}},
        },
    },
)
async def health():
    postgres = check_postgres()
    ollama = await check_ollama()
    status = "ok" if postgres == "up" and ollama == "up" else "degraded"
    return {
        "status": status,
        "postgres": postgres,
        "ollama": ollama,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
