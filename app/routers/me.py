from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import HealthGoal, MacroTargetsView, MeMacrosResponse
from app.openapi_responses import with_ac_baseline
from app.services import jwt_decoder, profile_service, user_preferences_service
from app.services.nutrition_engine import (
    IncompleteProfile,
    build_user_profile,
    compute_macro_targets,
    compute_tdee,
)
from app.services.user_preferences_service import PreferencesUpdate, PreferencesView

router = APIRouter(prefix="/me")


def _user_id_from_auth(authorization: str | None) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    identity = jwt_decoder.decode(authorization.removeprefix("Bearer "))
    return identity.user_id


_DESCRIPTION = """
Retourne la cible journaliere de l'utilisateur authentifie (TDEE et macros).

Le TDEE est calcule via la formule de Mifflin-St Jeor a partir des champs
biometriques du profil (`gender`, `age`, `weight_kg`, `height_cm`, `activity_level`)
et de l'objectif sante (`health_goal`). La repartition macro suit l'objectif :

- `balance` : 50 % glucides, 20 % proteines, 30 % lipides
- `weight_loss` : 40 / 30 / 30
- `muscle_gain` : 45 / 30 / 25
- `sport_performance` : 55 / 20 / 25

Quand un champ biometrique est manquant, la reponse renseigne
`profile_completion_required=true` et liste les `missing_fields` plutot que de
retourner une cible degradee.

**Authentification** : header `Authorization: Bearer <jwt>` requis.
"""

_MACROS_EXAMPLES = {
    "complete_profile": {
        "summary": "Profil complet : TDEE et macros calcules",
        "value": {
            "profile_completion_required": False,
            "missing_fields": [],
            "tdee": 2450,
            "macros": {
                "calories": 2450,
                "protein_g": 184.0,
                "carbs_g": 276.0,
                "fat_g": 68.0,
            },
        },
    },
    "incomplete_profile": {
        "summary": "Profil incomplet : champs manquants",
        "value": {
            "profile_completion_required": True,
            "missing_fields": ["weight_kg", "height_cm"],
            "tdee": None,
            "macros": None,
        },
    },
}


@router.get(
    "/macros",
    response_model=MeMacrosResponse,
    tags=["Profil"],
    summary="Cible journaliere TDEE et macros pour l'utilisateur",
    description=_DESCRIPTION,
    responses=with_ac_baseline(
        {
            200: {
                "description": "Cible journaliere ou liste des champs manquants.",
                "content": {"application/json": {"examples": _MACROS_EXAMPLES}},
            },
        },
    ),
)
def get_my_macros(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> MeMacrosResponse:
    user_id = _user_id_from_auth(authorization)
    record = profile_service.get_profile(user_id, db)

    profile_or_missing = build_user_profile(record)
    if isinstance(profile_or_missing, IncompleteProfile):
        return MeMacrosResponse(
            profile_completion_required=True,
            missing_fields=profile_or_missing.missing_fields,
        )

    # health_goal n'est pas dans la biometrie ; on retombe sur balance par defaut
    # plutot que d'exiger une valeur explicite (objectif sante = preference, pas
    # condition de calcul du TDEE).
    health_goal = (
        HealthGoal(record.health_goal)
        if record is not None and record.health_goal is not None
        else HealthGoal.balance
    )
    tdee = compute_tdee(profile_or_missing)
    macros = compute_macro_targets(tdee, health_goal)

    return MeMacrosResponse(
        profile_completion_required=False,
        missing_fields=[],
        tdee=tdee,
        macros=MacroTargetsView(
            calories=macros.calories,
            protein_g=macros.protein_g,
            carbs_g=macros.carbs_g,
            fat_g=macros.fat_g,
        ),
    )


@router.get(
    "/preferences",
    response_model=PreferencesView,
    tags=["Profil"],
    summary="Preferences LLM de l'utilisateur",
)
def get_my_preferences(
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> PreferencesView:
    user_id = _user_id_from_auth(authorization)
    return user_preferences_service.get_preferences(user_id, db)


@router.patch(
    "/preferences",
    response_model=PreferencesView,
    tags=["Profil"],
    summary="Met a jour les preferences LLM de l'utilisateur",
)
def patch_my_preferences(
    payload: PreferencesUpdate,
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> PreferencesView:
    user_id = _user_id_from_auth(authorization)
    return user_preferences_service.update_preferences(user_id, db, payload)
