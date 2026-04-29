from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import NutritionGoal
from app.models.schemas import NutritionGoalRequest


def get_profile(user_id: int, db: Session) -> NutritionGoal | None:
    return db.query(NutritionGoal).filter(NutritionGoal.user_id == user_id).one_or_none()


def upsert_profile(
    user_id: int, payload: NutritionGoalRequest, db: Session
) -> NutritionGoal:
    profile = get_profile(user_id, db)
    fields = payload.model_dump(exclude_unset=False)
    if profile is None:
        profile = NutritionGoal(user_id=user_id, **_normalize(fields))
        db.add(profile)
    else:
        for key, value in _normalize(fields).items():
            setattr(profile, key, value)
    db.commit()
    db.refresh(profile)
    return profile


def _normalize(fields: dict) -> dict:
    # Pydantic serialise l'enum HealthGoal en str via model_dump(mode="python") par defaut,
    # mais on stocke la valeur de l'enum (str) dans la colonne VARCHAR(30).
    health_goal = fields.get("health_goal")
    if health_goal is not None and hasattr(health_goal, "value"):
        fields["health_goal"] = health_goal.value
    return fields
