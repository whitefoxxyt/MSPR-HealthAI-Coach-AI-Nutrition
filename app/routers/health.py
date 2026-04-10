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


@router.get("/health")
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
