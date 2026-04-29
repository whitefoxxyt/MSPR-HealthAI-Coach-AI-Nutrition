from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import NutritionGoalRequest, NutritionGoalResponse
from app.services import jwt_decoder, profile_service

router = APIRouter(prefix="/nutrition-goals", tags=["nutrition-goals"])


def _user_id_from_auth(authorization: str | None) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    identity = jwt_decoder.decode(authorization.removeprefix("Bearer "))
    try:
        return int(identity.user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Sujet JWT invalide.")


@router.get("/me", response_model=NutritionGoalResponse)
def get_my_profile(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> NutritionGoalResponse:
    user_id = _user_id_from_auth(authorization)
    profile = profile_service.get_profile(user_id, db)
    if profile is None:
        raise HTTPException(status_code=404, detail="Profil nutritionnel non configure.")
    return NutritionGoalResponse.model_validate(profile)


@router.put("/me", response_model=NutritionGoalResponse)
def upsert_my_profile(
    payload: NutritionGoalRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> NutritionGoalResponse:
    user_id = _user_id_from_auth(authorization)
    profile = profile_service.upsert_profile(user_id, payload, db)
    return NutritionGoalResponse.model_validate(profile)
