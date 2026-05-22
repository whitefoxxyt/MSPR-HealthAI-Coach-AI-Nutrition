from __future__ import annotations

from fastapi import APIRouter, Body, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import NutritionGoalRequest, NutritionGoalResponse
from app.openapi_responses import with_ac_baseline
from app.services import jwt_decoder, profile_service

router = APIRouter(prefix="/nutrition-goals")


def _user_id_from_auth(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    identity = jwt_decoder.decode(authorization.removeprefix("Bearer "))
    return identity.user_id


_GET_DESCRIPTION = """
Renvoie le profil nutritionnel de l'utilisateur authentifie.

Le profil regroupe l'objectif sante (`weight_loss`, `muscle_gain`, `balance`, `sport_performance`), les cibles caloriques et macros, les allergies declarees et le regime (`omnivore`, `vegetarien`, `vegan`, `sans_gluten`).

Renvoie 404 si aucun profil n'a encore ete configure (un PUT initial est alors necessaire).

**Authentification** : header `Authorization: Bearer <jwt>` requis. Le profil est strictement scope a l'utilisateur du JWT.
"""

_PUT_DESCRIPTION = """
Cree ou met a jour le profil nutritionnel de l'utilisateur authentifie (upsert).

Tous les champs sont optionnels : un PUT partiel ecrase la ligne avec la valeur fournie. Pour conserver une valeur existante, le client doit la renvoyer explicitement.

Les valeurs sont utilisees ensuite par `/analyze-meal` (detection des desequilibres) et `/generate-meal-plan` (resolution de l'objectif sante par defaut).

**Authentification** : header `Authorization: Bearer <jwt>` requis.
"""

_GOAL_RESPONSE_EXAMPLE = {
    "user_id": 42,
    "health_goal": "muscle_gain",
    "calories_target": 2400,
    "protein_g": "140.0",
    "carbs_g": "260.0",
    "fat_g": "80.0",
    "allergies": ["arachides", "fruits a coque"],
    "diet_type": "omnivore",
}

_GOAL_REQUEST_EXAMPLES = {
    "muscle_gain": {
        "summary": "Prise de masse, omnivore",
        "value": {
            "health_goal": "muscle_gain",
            "calories_target": 2400,
            "protein_g": 140,
            "carbs_g": 260,
            "fat_g": 80,
            "allergies": ["arachides"],
            "diet_type": "omnivore",
        },
    },
    "weight_loss_vegetarien": {
        "summary": "Perte de poids, vegetarien",
        "value": {
            "health_goal": "weight_loss",
            "calories_target": 1700,
            "protein_g": 90,
            "carbs_g": 170,
            "fat_g": 55,
            "allergies": [],
            "diet_type": "vegetarien",
        },
    },
    "minimal": {
        "summary": "Profil minimal (objectif sante seul)",
        "value": {"health_goal": "balance"},
    },
}


@router.get(
    "/me",
    response_model=NutritionGoalResponse,
    tags=["Profil"],
    summary="Recupere le profil nutritionnel de l'utilisateur",
    description=_GET_DESCRIPTION,
    responses=with_ac_baseline(
        {
            200: {
                "description": "Profil nutritionnel courant.",
                "content": {"application/json": {"example": _GOAL_RESPONSE_EXAMPLE}},
            },
            404: {
                "description": "Aucun profil nutritionnel configure (faire un PUT initial)."
            },
        },
    ),
)
def get_my_profile(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> NutritionGoalResponse:
    user_id = _user_id_from_auth(authorization)
    profile = profile_service.get_profile(user_id, db)
    if profile is None:
        raise HTTPException(
            status_code=404, detail="Profil nutritionnel non configure."
        )
    return NutritionGoalResponse.model_validate(profile)


@router.put(
    "/me",
    response_model=NutritionGoalResponse,
    tags=["Profil"],
    summary="Cree ou met a jour le profil nutritionnel (upsert)",
    description=_PUT_DESCRIPTION,
    responses=with_ac_baseline(
        {
            200: {
                "description": "Profil mis a jour (ou cree).",
                "content": {"application/json": {"example": _GOAL_RESPONSE_EXAMPLE}},
            },
            422: {
                "description": "Payload invalide (objectif sante inconnu, valeurs non numeriques, etc)."
            },
        },
    ),
)
def upsert_my_profile(
    payload: NutritionGoalRequest = Body(..., openapi_examples=_GOAL_REQUEST_EXAMPLES),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> NutritionGoalResponse:
    user_id = _user_id_from_auth(authorization)
    profile = profile_service.upsert_profile(user_id, payload, db)
    return NutritionGoalResponse.model_validate(profile)
