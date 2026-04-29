from __future__ import annotations

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.db.models import MealAnalysis
from app.models.schemas import MealAnalysisItem, PaginatedAnalysesResponse


def list_user_analyses(
    user_id: int, limit: int, offset: int, db: Session
) -> PaginatedAnalysesResponse:
    base = db.query(MealAnalysis).filter(MealAnalysis.user_id == user_id)
    total = base.with_entities(func.count(MealAnalysis.id)).scalar() or 0
    rows = (
        base.with_entities(
            MealAnalysis.id,
            MealAnalysis.detected_foods,
            MealAnalysis.macros,
            MealAnalysis.recommendations,
            MealAnalysis.created_at,
        )
        .order_by(MealAnalysis.created_at.desc())
        .limit(limit)
        .offset(offset)
        .all()
    )
    items = [
        MealAnalysisItem(
            id=row.id,
            detected_foods=row.detected_foods or [],
            macros=row.macros or {},
            recommendations=row.recommendations or [],
            created_at=row.created_at,
        )
        for row in rows
    ]
    return PaginatedAnalysesResponse(items=items, total=total, limit=limit, offset=offset)
