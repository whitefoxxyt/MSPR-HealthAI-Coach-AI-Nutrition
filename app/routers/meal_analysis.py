from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import PaginatedAnalysesResponse
from app.services import jwt_decoder, meal_analysis_history
from app.services.meal_analysis_orchestrator import analyze_meal as orchestrate

router = APIRouter()

_MAX_SIZE = 10 * 1024 * 1024  # 10 Mo
_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _user_id_from_auth(authorization: str | None) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    identity = jwt_decoder.decode(authorization.removeprefix("Bearer "))
    try:
        return int(identity.user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Sujet JWT invalide.") from exc


@router.post("/analyze-meal")
async def analyze_meal(
    photo: UploadFile = File(...),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    if photo.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Type non supporte : {photo.content_type}. Utilisez JPEG, PNG ou WebP.",
        )
    image_bytes = await photo.read()
    if len(image_bytes) > _MAX_SIZE:
        raise HTTPException(status_code=413, detail="Image trop volumineuse (max 10 Mo).")

    user_id = _user_id_from_auth(authorization)
    return await orchestrate(image_bytes, user_id, db)


@router.get("/meal-analyses/me", response_model=PaginatedAnalysesResponse)
def list_my_meal_analyses(
    authorization: str | None = Header(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedAnalysesResponse:
    user_id = _user_id_from_auth(authorization)
    return meal_analysis_history.list_user_analyses(user_id, limit, offset, db)
