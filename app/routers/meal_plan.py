from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.limiter import limiter
from app.models.schemas import MealPlanRequest, MealPlanResponse, PaginatedPlansResponse
from app.services import jwt_decoder, meal_plan_history, meal_plan_orchestrator
from app.services.decrim_retry_orchestrator import InfeasibleConstraintsError

# Note : pas de `from __future__ import annotations` ici. Slowapi enveloppe la
# fonction et FastAPI doit pouvoir resoudre les types Pydantic au moment de
# l'inspection. Avec l'import differe, MealPlanRequest reste un ForwardRef et
# le decodage du body echoue (PydanticUserError "class-not-fully-defined").

router = APIRouter()


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


_PLAN_DESCRIPTION = """
Genere un plan repas personnalise sur 1 a 30 jours en fonction de l'objectif sante, du regime, des allergies et du budget.

Pipeline applique :

1. Resolution de l'objectif sante : valeur explicite > profil utilisateur (`nutrition-goals/me`) > `balance` par defaut.
2. Lecture des entitlements (tier utilisateur via MSPR-AUTH). Les tiers `premium` et `premium_plus` bypassent le cache et regenerent systematiquement.
3. Cache contenu : hash canonique des inputs (objectif, regime, allergies triees, budget, calories cible, duree). Cache hit, on renvoie le plan stocke.
4. Cache miss : appel Ollama (Gemma3:4b) avec prompt structure et `format: json`. En cas d'echec ou JSON invalide, repli sur la matrice 16 plans statiques (`fallback: true`).
5. Persistance dans `meal_plans` et renvoi du plan complet.

**Rate limit** : 10 requetes par heure et 3 par minute par utilisateur (header standard `Retry-After` en cas de depassement).

**Latence** : 5-30 s sur CPU pour une generation LLM. Cache hit < 100 ms.

**Authentification** : header `Authorization: Bearer <jwt>` requis. Le JWT est aussi propage a MSPR-AUTH pour resoudre les entitlements.
"""

_PLAN_REQUEST_EXAMPLES = {
    "muscle_gain_omnivore": {
        "summary": "Prise de masse, omnivore, 7 jours",
        "value": {
            "health_goal": "muscle_gain",
            "diet_type": "omnivore",
            "duration_days": 7,
            "allergies": [],
            "budget_eur_per_day": 15.0,
        },
    },
    "weight_loss_vegetarien": {
        "summary": "Perte de poids, vegetarien, 14 jours",
        "value": {
            "health_goal": "weight_loss",
            "diet_type": "vegetarien",
            "duration_days": 14,
            "allergies": ["arachides"],
            "budget_eur_per_day": 10.0,
        },
    },
    "balance_vegan_minimum": {
        "summary": "Equilibre, vegan, requete minimale (1 jour, sans budget)",
        "value": {"diet_type": "vegan", "duration_days": 1},
    },
}

_PLAN_RESPONSE_EXAMPLE = {
    "plan_id": 42,
    "fallback": False,
    "days": [
        {
            "day": 1,
            "meals": [
                {
                    "name": "Bowl quinoa, poulet et legumes verts",
                    "macros": {
                        "calories": 620,
                        "protein_g": 45.0,
                        "carbs_g": 65.0,
                        "fat_g": 18.0,
                    },
                    "ingredients": [
                        "100 g quinoa",
                        "150 g blanc de poulet",
                        "200 g brocolis",
                        "1 c.s. huile d'olive",
                    ],
                    "est_budget_eur": 5.5,
                    "prep_time_min": 25,
                }
            ],
        }
    ],
}


@router.post(
    "/generate-meal-plan",
    response_model=MealPlanResponse,
    tags=["Plans"],
    summary="Genere un plan repas personnalise (1 a 30 jours)",
    description=_PLAN_DESCRIPTION,
    responses={
        200: {
            "description": "Plan repas genere ou recupere du cache. `fallback: true` indique un repli matrice statique.",
            "content": {"application/json": {"example": _PLAN_RESPONSE_EXAMPLE}},
        },
        401: {"description": "JWT manquant, malforme ou invalide."},
        422: {
            "description": "Payload invalide (regime inconnu, duree hors [1, 30], etc)."
        },
        429: {"description": "Rate limit depasse (10/heure ou 3/minute)."},
        503: {
            "description": (
                "Contraintes infaisables (allergies / regime impossibles a satisfaire) "
                "ou Ollama injoignable et fallback statique indisponible."
            )
        },
    },
)
@limiter.limit("10/hour;3/minute")
async def generate_meal_plan(
    request: Request,  # requis par SlowAPI pour key_func
    payload: MealPlanRequest = Body(..., openapi_examples=_PLAN_REQUEST_EXAMPLES),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> MealPlanResponse:
    user_id, token = _auth(authorization)
    try:
        return await meal_plan_orchestrator.generate(user_id, payload, token, db)
    except InfeasibleConstraintsError:
        # DeCRIM-light a epuise les retries et le plan statique de fallback
        # viole encore les allergies / regime : on signale au client que la
        # combinaison demandee est infaisable plutot que de servir un plan
        # potentiellement dangereux. Body JSON plat : detail + infeasible cote
        # racine, contrat explicite pour le client.
        return JSONResponse(
            status_code=503,
            content={
                "detail": (
                    "Contraintes infaisables, ajustez vos contraintes "
                    "(allergies / regime)."
                ),
                "infeasible": True,
            },
        )


_PLANS_HISTORY_DESCRIPTION = """
Renvoie l'historique pagine des plans repas generes pour l'utilisateur authentifie.

Tri decroissant par date de generation. Chaque element contient l'objectif sante, les contraintes (regime, allergies, budget, duree, calories cible) et le plan complet (jours et repas).

Pagination par `limit` (defaut 20, max 100) et `offset` (defaut 0). Le total est inclus dans la reponse pour permettre au client d'afficher un compteur ou de preparer une pagination infinie.

**Authentification** : header `Authorization: Bearer <jwt>` requis. L'historique est strictement scope a l'utilisateur du JWT.
"""

_PLANS_HISTORY_RESPONSE_EXAMPLE = {
    "items": [
        {
            "id": 42,
            "objective": "muscle_gain",
            "constraints": {
                "diet_type": "omnivore",
                "allergies": ["arachides"],
                "duration_days": 7,
                "budget_per_day": 15.0,
            },
            "plan": {
                "fallback": False,
                "days": [
                    {
                        "day": 1,
                        "meals": [
                            {
                                "name": "Bowl quinoa, poulet et legumes verts",
                                "macros": {
                                    "calories": 620,
                                    "protein_g": 45.0,
                                    "carbs_g": 65.0,
                                    "fat_g": 18.0,
                                },
                                "ingredients": [
                                    "100 g quinoa",
                                    "150 g blanc de poulet",
                                ],
                                "est_budget_eur": 5.5,
                                "prep_time_min": 25,
                            }
                        ],
                    }
                ],
            },
            "generated_at": "2026-04-29T10:15:00",
        }
    ],
    "total": 12,
    "limit": 20,
    "offset": 0,
}


@router.get(
    "/meal-plans/me",
    response_model=PaginatedPlansResponse,
    tags=["Historique"],
    summary="Historique pagine des plans repas de l'utilisateur",
    description=_PLANS_HISTORY_DESCRIPTION,
    responses={
        200: {
            "description": "Liste pagine de plans repas (tri decroissant sur generated_at).",
            "content": {
                "application/json": {"example": _PLANS_HISTORY_RESPONSE_EXAMPLE}
            },
        },
        401: {"description": "JWT manquant, malforme ou invalide."},
    },
)
def list_my_meal_plans(
    authorization: str | None = Header(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedPlansResponse:
    user_id, _ = _auth(authorization)
    return meal_plan_history.list_user_plans(user_id, limit, offset, db)
