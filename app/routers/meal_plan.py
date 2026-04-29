from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.limiter import limiter
from app.models.schemas import MealPlanRequest, MealPlanResponse, PaginatedPlansResponse
from app.services import jwt_decoder, meal_plan_history, meal_plan_orchestrator

# Note : pas de `from __future__ import annotations` ici. Slowapi enveloppe la
# fonction et FastAPI doit pouvoir resoudre les types Pydantic au moment de
# l'inspection. Avec l'import differe, MealPlanRequest reste un ForwardRef et
# le decodage du body echoue (PydanticUserError "class-not-fully-defined").

router = APIRouter(tags=["meal-plan"])


def _auth(authorization: str | None) -> tuple[int, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    token = authorization.removeprefix("Bearer ")
    identity = jwt_decoder.decode(token)
    try:
        user_id = int(identity.user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Sujet JWT invalide.") from exc
    return user_id, token


@router.post("/generate-meal-plan", response_model=MealPlanResponse)
@limiter.limit("10/hour;3/minute")
async def generate_meal_plan(
    request: Request,  # requis par SlowAPI pour key_func
    payload: MealPlanRequest,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> MealPlanResponse:
    user_id, token = _auth(authorization)
    return await meal_plan_orchestrator.generate(user_id, payload, token, db)


@router.get("/meal-plans/me", response_model=PaginatedPlansResponse)
def list_my_meal_plans(
    authorization: str | None = Header(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedPlansResponse:
    user_id, _ = _auth(authorization)
    return meal_plan_history.list_user_plans(user_id, limit, offset, db)
