from __future__ import annotations

from decimal import Decimal

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models.schemas import (
    HealthGoal,
    MealPlanRequest,
    MealPlanResponse,
    PlanInputs,
)
from app.services.entitlements_client import get_entitlements
from app.services.fallback_loader import load_fallback_plan
from app.services.llm_client import compute_inputs_hash, generate_plan
from app.services.profile_service import get_profile
from app.services.user_preferences_service import get_preferences

# Tiers entitled au bypass cache : chaque appel re-genere via Ollama.
_PREMIUM_TIERS = frozenset({"premium", "premium_plus"})


async def generate(
    user_id: int,
    request: MealPlanRequest,
    jwt: str,
    db: Session,
) -> MealPlanResponse:
    """Genere un plan repas en composant llm_client + entitlements + profile.

    - Resout health_goal : explicite > profil > balance.
    - Bypass cache si tier in {premium, premium_plus} (issue NUT-5).
    - Delegue cache + retry + DeCRIM-light + persistance a llm_client.
    - InfeasibleConstraintsError remontee : le router traduit en HTTP 503.
    """
    health_goal = _resolve_health_goal(request.health_goal, user_id, db)

    entitlements = await get_entitlements(str(user_id), jwt)
    bypass_cache = entitlements.tier in _PREMIUM_TIERS

    plan_inputs = _build_inputs(user_id, request, health_goal)
    backend = get_preferences(user_id, db).effective_llm.value

    plan, compliance_status, compliance_warnings = await generate_plan(
        plan_inputs,
        db,
        bypass_cache=bypass_cache,
        fallback_loader=load_fallback_plan,
        primary_backend=backend,
    )

    plan_id = _latest_plan_id(db, user_id, compute_inputs_hash(plan_inputs))
    db.commit()

    return MealPlanResponse(
        plan_id=plan_id,
        fallback=plan.fallback,
        days=plan.days,
        compliance_status=compliance_status.value,
        compliance_warnings=compliance_warnings,
    )


def _resolve_health_goal(
    explicit: HealthGoal | None, user_id: int, db: Session
) -> HealthGoal:
    if explicit is not None:
        return explicit
    profile = get_profile(user_id, db)
    if profile is not None and profile.health_goal:
        return HealthGoal(profile.health_goal)
    return HealthGoal.balance


def _build_inputs(
    user_id: int, request: MealPlanRequest, health_goal: HealthGoal
) -> PlanInputs:
    budget = (
        Decimal(str(request.budget_eur_per_day))
        if request.budget_eur_per_day is not None
        else None
    )
    return PlanInputs(
        user_id=user_id,
        objective=health_goal.value,
        duration_days=request.duration_days,
        diet_type=request.diet_type.value,
        allergies=request.allergies,
        budget_per_day=budget,
    )


def _latest_plan_id(db: Session, user_id: int, inputs_hash: str) -> int:
    """Recupere l'id du plan que generate_plan vient de persister ou de servir du cache.

    Pas de filtre par backend : si le fallback chain a kick in (Mistral KO ->
    Ollama OK), la ligne porte llm_backend_used du secondaire effectivement
    utilise alors que le caller connait le primaire demande. La cle naturelle
    "latest pour (user, hash)" suffit puisque generate_plan vient juste de
    persister cette ligne.
    """
    row = db.execute(
        text(
            "SELECT id FROM meal_plans "
            "WHERE user_id = :uid AND inputs_hash = :h "
            "ORDER BY generated_at DESC LIMIT 1"
        ),
        {"uid": user_id, "h": inputs_hash},
    ).fetchone()
    if row is None:
        # generate_plan persiste systematiquement (success / fallback / cache hit).
        # Si on arrive ici c'est une erreur de logique amont.
        raise RuntimeError("plan persiste introuvable apres generate_plan")
    return int(row.id)
