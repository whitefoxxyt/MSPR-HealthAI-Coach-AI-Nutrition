from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import MealPlan
from app.models.schemas import MealPlanHistoryItem, PaginatedPlansResponse


def list_user_plans(
    user_id: str, limit: int, offset: int, db: Session
) -> PaginatedPlansResponse:
    base = db.query(MealPlan).filter(MealPlan.user_id == user_id)
    total = base.with_entities(func.count(MealPlan.id)).scalar() or 0
    rows = (
        base.with_entities(
            MealPlan.id,
            MealPlan.objective,
            MealPlan.constraints,
            MealPlan.plan,
            MealPlan.generated_at,
        )
        .order_by(MealPlan.generated_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    items = [
        MealPlanHistoryItem(
            id=row.id,
            objective=row.objective,
            constraints=row.constraints or {},
            plan=row.plan or {},
            generated_at=row.generated_at,
        )
        for row in rows
    ]
    return PaginatedPlansResponse(items=items, total=total, limit=limit, offset=offset)
