from __future__ import annotations

from enum import Enum

from sqlalchemy.orm import Session

from app.db.models import NutritionGoal
from app.models.schemas import NutritionGoalRequest


def get_profile(user_id: str, db: Session) -> NutritionGoal | None:
    return db.query(NutritionGoal).filter(NutritionGoal.user_id == user_id).one_or_none()


def upsert_profile(
    user_id: str, payload: NutritionGoalRequest, db: Session
) -> NutritionGoal:
    profile = get_profile(user_id, db)
    fields = _normalize(payload.model_dump())
    if profile is None:
        profile = NutritionGoal(user_id=user_id, **fields)
        db.add(profile)
    else:
        for key, value in fields.items():
            setattr(profile, key, value)
    db.commit()
    db.refresh(profile)
    return profile


def _normalize(fields: dict) -> dict:
    # Pydantic serialise les enums en instances via model_dump(mode="python") par defaut.
    # On extrait .value pour stocker le string brut dans les colonnes VARCHAR
    # (health_goal, gender, activity_level, diet_type, ...).
    return {
        key: value.value if isinstance(value, Enum) else value
        for key, value in fields.items()
    }
