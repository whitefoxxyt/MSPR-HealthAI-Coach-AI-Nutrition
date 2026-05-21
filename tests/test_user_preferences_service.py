from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.services.user_preferences_service import (
    PreferencesUpdate,
    get_preferences,
    update_preferences,
)


@pytest.fixture(autouse=True)
def _default_env_mistral(monkeypatch: pytest.MonkeyPatch) -> None:
    """Defaut env stable pour les tests : mistral, comme en prod."""
    monkeypatch.setattr(settings, "default_llm", "mistral")


def test_get_preferences_without_profile_returns_env_default(
    db_session: Session,
) -> None:
    view = get_preferences(user_id=1, db=db_session)

    assert view.preferred_llm is None
    assert view.effective_llm == "mistral"


def test_get_preferences_returns_user_pref_when_set(
    db_session: Session,
) -> None:
    db_session.execute(
        text(
            "INSERT INTO nutrition_goals (user_id, preferred_llm) "
            "VALUES (2, 'ollama')"
        )
    )
    db_session.commit()

    view = get_preferences(user_id=2, db=db_session)

    # La preference user prime sur le defaut env (mistral).
    assert view.preferred_llm == "ollama"
    assert view.effective_llm == "ollama"


def test_update_preferences_creates_profile_when_none_exists(
    db_session: Session,
) -> None:
    view = update_preferences(
        user_id=3,
        db=db_session,
        prefs=PreferencesUpdate(preferred_llm="ollama"),
    )

    assert view.preferred_llm == "ollama"
    assert view.effective_llm == "ollama"
    # La ligne nutrition_goals est creee meme si seule la preference est definie.
    row = db_session.execute(
        text("SELECT preferred_llm FROM nutrition_goals WHERE user_id = 3")
    ).fetchone()
    assert row is not None
    assert row.preferred_llm == "ollama"


def test_update_preferences_overwrites_existing_value(
    db_session: Session,
) -> None:
    update_preferences(
        user_id=4, db=db_session, prefs=PreferencesUpdate(preferred_llm="ollama")
    )
    view = update_preferences(
        user_id=4, db=db_session, prefs=PreferencesUpdate(preferred_llm="mistral")
    )

    assert view.preferred_llm == "mistral"
    assert view.effective_llm == "mistral"


def test_update_preferences_with_none_resets_to_env_default(
    db_session: Session,
) -> None:
    update_preferences(
        user_id=5, db=db_session, prefs=PreferencesUpdate(preferred_llm="ollama")
    )
    view = update_preferences(
        user_id=5, db=db_session, prefs=PreferencesUpdate(preferred_llm=None)
    )

    # Reset : preferred_llm est NULL en BDD, effective_llm tombe sur le defaut env.
    assert view.preferred_llm is None
    assert view.effective_llm == "mistral"
    row = db_session.execute(
        text("SELECT preferred_llm FROM nutrition_goals WHERE user_id = 5")
    ).fetchone()
    assert row.preferred_llm is None
