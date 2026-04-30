from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import HealthGoal, MacroTargetsView, MeMacrosResponse
from app.services import jwt_decoder, profile_service
from app.services.nutrition_engine import (
    IncompleteProfile,
    build_user_profile,
    compute_macro_targets,
    compute_tdee,
)

router = APIRouter(prefix="/me")


def _user_id_from_auth(authorization: str | None) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    identity = jwt_decoder.decode(authorization.removeprefix("Bearer "))
    try:
        return int(identity.user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Sujet JWT invalide.") from exc


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


@router.get(
    "/macros",
    response_model=MeMacrosResponse,
    tags=["Profil"],
    summary="Cible journaliere TDEE et macros pour l'utilisateur",
    description=_DESCRIPTION,
    responses={
        200: {"description": "Cible journaliere ou liste des champs manquants."},
        401: {"description": "JWT manquant, malforme ou invalide."},
    },
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
