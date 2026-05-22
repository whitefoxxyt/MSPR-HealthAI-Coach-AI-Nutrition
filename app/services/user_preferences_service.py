from __future__ import annotations

from enum import Enum

from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import NutritionGoal


class LLMBackend(str, Enum):
    """Backends LLM supportes (PRD #71). Toute autre valeur est rejetee."""

    ollama = "ollama"
    mistral = "mistral"


class PreferencesUpdate(BaseModel):
    """Payload PATCH /me/preferences. preferred_llm=null reset au defaut env."""

    preferred_llm: LLMBackend | None = None


class PreferencesView(BaseModel):
    """Reponse GET/PATCH /me/preferences.

    preferred_llm : valeur brute persistee pour cet utilisateur (None si jamais
                    definie ou explicitement reset).
    effective_llm : backend effectivement utilise par l'orchestrator
                    (preference user OR settings.default_llm).
    """

    preferred_llm: LLMBackend | None = None
    effective_llm: LLMBackend


def get_preferences(user_id: str, db: Session) -> PreferencesView:
    profile = _get_profile(user_id, db)
    raw = profile.preferred_llm if profile is not None else None
    return _build_view(raw)


def update_preferences(
    user_id: str, db: Session, prefs: PreferencesUpdate
) -> PreferencesView:
    """Upsert de la preference utilisateur.

    Cree la ligne nutrition_goals si elle n'existe pas (seule la preference
    LLM est definie, le reste du profil reste NULL). Idempotent : meme appel
    deux fois -> meme etat final.
    """
    new_value = prefs.preferred_llm.value if prefs.preferred_llm is not None else None
    profile = _get_profile(user_id, db)
    if profile is None:
        profile = NutritionGoal(user_id=user_id, preferred_llm=new_value)
        db.add(profile)
    else:
        profile.preferred_llm = new_value
    db.commit()
    return _build_view(new_value)


def _get_profile(user_id: str, db: Session) -> NutritionGoal | None:
    return (
        db.query(NutritionGoal).filter(NutritionGoal.user_id == user_id).one_or_none()
    )


def _build_view(raw: str | None) -> PreferencesView:
    return PreferencesView(
        preferred_llm=LLMBackend(raw) if raw is not None else None,
        effective_llm=LLMBackend(raw or settings.default_llm),
    )
