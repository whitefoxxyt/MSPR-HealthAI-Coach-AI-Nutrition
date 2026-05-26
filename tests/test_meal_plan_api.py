from __future__ import annotations

import json
import re
from collections.abc import Callable, Generator
from typing import Any

import httpx
import pytest
import respx
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import get_db
from app.main import app
from app.services import entitlements_client
from app.services.entitlements_client import Entitlements
from tests.conftest import TEST_AUTH_SECRET, TEST_OLLAMA_HOST


def _ollama_payload(plan: dict[str, Any]) -> dict[str, Any]:
    return {"response": json.dumps(plan), "done": True}


def _valid_plan(marker_calories: int = 450) -> dict[str, Any]:
    return {
        "fallback": False,
        "days": [
            {
                "day": 1,
                "meals": [
                    {
                        "name": "Petit-dejeuner protein",
                        "macros": {
                            "calories": marker_calories,
                            "protein_g": 30.0,
                            "carbs_g": 40.0,
                            "fat_g": 15.0,
                        },
                        "ingredients": ["avoine", "tofu"],
                        "est_budget_eur": 2.5,
                        "prep_time_min": 10,
                    }
                ],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "better_auth_secret", TEST_AUTH_SECRET)
    monkeypatch.setattr(settings, "ollama_host", TEST_OLLAMA_HOST)
    entitlements_client._cache.clear()
    entitlements_client._stale.clear()


@pytest.fixture(autouse=True)
def _entitlements_free(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """Par defaut : tier free. Les tests qui veulent premium appellent _set_tier."""
    from app.services import meal_plan_orchestrator

    holder: dict[str, str] = {"tier": "free"}

    async def _fake_get(user_id: str, jwt: str) -> Entitlements:  # noqa: ARG001
        return Entitlements(tier=holder["tier"], expires_at=None, features=())  # type: ignore[arg-type]

    monkeypatch.setattr(entitlements_client, "get_entitlements", _fake_get)
    monkeypatch.setattr(meal_plan_orchestrator, "get_entitlements", _fake_get)

    def _set_tier(tier: str) -> None:
        holder["tier"] = tier

    return _set_tier


@pytest.fixture
def set_tier(_entitlements_free: Callable[[str], None]) -> Callable[[str], None]:
    return _entitlements_free


@pytest.fixture(autouse=True)
def _reset_limiter() -> None:
    """Reset SlowAPI in-memory storage entre tests pour eviter la contamination."""
    from app.limiter import limiter

    limiter.reset()


@pytest.fixture
def client(db_session: Session) -> Generator[TestClient, None, None]:
    def _override_get_db() -> Generator[Session, None, None]:
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.pop(get_db, None)


def test_generate_meal_plan_returns_200_with_plan_id_and_days(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
) -> None:
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="42")}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["plan_id"] >= 1
    assert body["fallback"] is False
    assert isinstance(body["days"], list)
    assert len(body["days"]) == 1
    assert body["days"][0]["day"] == 1
    assert body["days"][0]["meals"][0]["name"] == "Petit-dejeuner protein"


def test_free_tier_cache_hit_skips_second_ollama_call(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
) -> None:
    """Tier free + memes inputs : 2eme requete sert le cache, pas d'appel Ollama."""
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan(marker_calories=1234))
    )
    headers = {"Authorization": f"Bearer {valid_jwt(user_id="200")}"}
    body = {
        "health_goal": "balance",
        "diet_type": "omnivore",
        "duration_days": 1,
        "allergies": [],
        "budget_eur_per_day": 15,
    }

    first = client.post("/api/v1/generate-meal-plan", json=body, headers=headers)
    second = client.post("/api/v1/generate-meal-plan", json=body, headers=headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert route.call_count == 1
    # Le 2eme appel renvoie le plan cache (memes calories marker).
    assert second.json()["days"][0]["meals"][0]["macros"]["calories"] == 1234


def test_premium_tier_bypasses_cache_and_regenerates(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    set_tier: Callable[[str], None],
) -> None:
    """Tier premium : 2 appels memes inputs -> 2 generations distinctes."""
    set_tier("premium")
    plans = [_valid_plan(marker_calories=111), _valid_plan(marker_calories=222)]
    route = mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_payload(plans[0])),
            httpx.Response(200, json=_ollama_payload(plans[1])),
        ]
    )
    headers = {"Authorization": f"Bearer {valid_jwt(user_id="300")}"}
    body = {
        "health_goal": "balance",
        "diet_type": "omnivore",
        "duration_days": 1,
        "allergies": [],
        "budget_eur_per_day": 15,
    }

    first = client.post("/api/v1/generate-meal-plan", json=body, headers=headers)
    second = client.post("/api/v1/generate-meal-plan", json=body, headers=headers)

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    assert route.call_count == 2
    assert first.json()["days"][0]["meals"][0]["macros"]["calories"] == 111
    assert second.json()["days"][0]["meals"][0]["macros"]["calories"] == 222


def test_premium_plus_tier_also_bypasses_cache(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    set_tier: Callable[[str], None],
) -> None:
    set_tier("premium_plus")
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )
    headers = {"Authorization": f"Bearer {valid_jwt(user_id="301")}"}
    body = {
        "health_goal": "balance",
        "diet_type": "omnivore",
        "duration_days": 1,
        "allergies": [],
        "budget_eur_per_day": 15,
    }

    client.post("/api/v1/generate-meal-plan", json=body, headers=headers)
    client.post("/api/v1/generate-meal-plan", json=body, headers=headers)

    assert route.call_count == 2


def test_falls_back_to_static_plan_when_ollama_unavailable(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
) -> None:
    """Ollama down (503 sur tous les retries) : retombee sur les plans NUT-8."""
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        503, json={"error": "ollama down"}
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 7,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="400")}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is True
    # Le fallback NUT-8 pour balance / omnivore couvre 7 jours x 3 repas.
    assert len(body["days"]) == 7
    assert all(len(d["meals"]) == 3 for d in body["days"])


def test_invalid_llm_json_retries_then_falls_back(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
) -> None:
    """Ollama renvoie un JSON sans champs requis sur les 3 essais : fallback."""
    invalid = {"days": "not a list"}  # rejete par Pydantic
    route = mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_payload(invalid)),
            httpx.Response(200, json=_ollama_payload(invalid)),
            httpx.Response(200, json=_ollama_payload(invalid)),
        ]
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "weight_loss",
            "diet_type": "vegetarien",
            "duration_days": 7,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="410")}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is True
    assert route.call_count == 3  # 1 essai + 2 retries


def test_invalid_then_valid_llm_returns_success_without_fallback(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
) -> None:
    """Premier JSON invalide, retry produit JSON valide : pas de fallback."""
    invalid = {"days": "not a list"}
    valid = _valid_plan(marker_calories=777)
    route = mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_payload(invalid)),
            httpx.Response(200, json=_ollama_payload(valid)),
        ]
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="411")}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is False
    assert body["days"][0]["meals"][0]["macros"]["calories"] == 777
    assert route.call_count == 2


def test_health_goal_falls_back_to_profile_when_request_null(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    db_session: Session,
) -> None:
    """health_goal=null + profil "weight_loss" -> objective dans le prompt = weight_loss."""
    # Pre-cree un profil nutritionnel pour user 500.
    db_session.execute(
        text(
            "INSERT INTO nutrition_goals (user_id, health_goal, diet_type) "
            "VALUES (500, 'weight_loss', 'omnivore')"
        )
    )
    db_session.commit()

    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": None,
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="500")}"},
    )

    assert response.status_code == 200, response.text
    sent_prompt = json.loads(route.calls.last.request.content)["prompt"]
    assert "weight_loss" in sent_prompt
    # On verifie en BDD que l'objective persiste correspond au profil.
    objective = db_session.execute(
        text("SELECT objective FROM meal_plans WHERE user_id = '500'")
    ).scalar()
    assert objective == "weight_loss"


def test_health_goal_defaults_to_balance_when_no_profile(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    db_session: Session,
) -> None:
    """Pas de profil + health_goal=null : default 'balance'."""
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": None,
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="501")}"},
    )

    assert response.status_code == 200, response.text
    objective = db_session.execute(
        text("SELECT objective FROM meal_plans WHERE user_id = '501'")
    ).scalar()
    assert objective == "balance"


def test_explicit_health_goal_overrides_profile(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    db_session: Session,
) -> None:
    """health_goal explicite != null : profil ignore."""
    db_session.execute(
        text(
            "INSERT INTO nutrition_goals (user_id, health_goal, diet_type) "
            "VALUES (502, 'weight_loss', 'omnivore')"
        )
    )
    db_session.commit()

    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "muscle_gain",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="502")}"},
    )

    assert response.status_code == 200, response.text
    objective = db_session.execute(
        text("SELECT objective FROM meal_plans WHERE user_id = '502'")
    ).scalar()
    assert objective == "muscle_gain"


def test_rate_limit_returns_429_after_threshold(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    set_tier: Callable[[str], None],
) -> None:
    """11eme requete dans la fenetre depasse 10/heure (ou 3/min) -> 429."""
    # Tier premium pour bypasser le cache : sinon les requetes apres la 1ere
    # frappent le cache et la limite ne se declenche pas (selon implementation).
    # Le rate limit fastapi/slowapi compte les requetes peu importe le cache.
    set_tier("premium")
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )
    headers = {"Authorization": f"Bearer {valid_jwt(user_id="600")}"}
    body = {
        "health_goal": "balance",
        "diet_type": "omnivore",
        "duration_days": 1,
        "allergies": [],
        "budget_eur_per_day": 15,
    }

    statuses = [
        client.post("/api/v1/generate-meal-plan", json=body, headers=headers).status_code
        for _ in range(11)
    ]

    # Au moins une 429 dans les 11 requetes (3/minute declenche au 4e, ou 10/hour au 11e).
    assert 429 in statuses, statuses
    assert statuses[-1] == 429


def test_rate_limit_keys_by_user_id_not_ip(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    set_tier: Callable[[str], None],
) -> None:
    """Deux users distincts depuis la meme IP : leurs quotas sont independants."""
    set_tier("premium")
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )
    body = {
        "health_goal": "balance",
        "diet_type": "omnivore",
        "duration_days": 1,
        "allergies": [],
        "budget_eur_per_day": 15,
    }

    # User 700 : epuise sa limite de minute (10/minute, le 11e doit etre bloque).
    headers_700 = {"Authorization": f"Bearer {valid_jwt(user_id="700")}"}
    for _ in range(10):
        client.post("/api/v1/generate-meal-plan", json=body, headers=headers_700)
    blocked = client.post("/api/v1/generate-meal-plan", json=body, headers=headers_700)
    assert blocked.status_code == 429

    # User 701 : son quota n'est pas affecte.
    headers_701 = {"Authorization": f"Bearer {valid_jwt(user_id="701")}"}
    fresh = client.post("/api/v1/generate-meal-plan", json=body, headers=headers_701)
    assert fresh.status_code == 200, fresh.text


def test_successful_generation_persists_row_with_inputs_hash(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    db_session: Session,
) -> None:
    """Chaque generation reussie cree une ligne meal_plans avec inputs_hash rempli."""
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "muscle_gain",
            "diet_type": "vegetarien",
            "duration_days": 3,
            "allergies": ["arachides"],
            "budget_eur_per_day": 12,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="800")}"},
    )

    assert response.status_code == 200, response.text
    row = db_session.execute(
        text(
            "SELECT id, user_id, inputs_hash, plan, objective "
            "FROM meal_plans WHERE user_id = '800'"
        )
    ).fetchone()
    assert row is not None
    assert row.user_id == "800"
    assert row.inputs_hash is not None
    # SHA256 hex = 64 caracteres.
    assert len(row.inputs_hash) == 64
    assert row.objective == "muscle_gain"
    assert response.json()["plan_id"] == row.id


def test_fallback_generation_also_persists_row(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    db_session: Session,
) -> None:
    """Le mode degrade insere aussi une ligne meal_plans (avec fallback=true)."""
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(503, json={"error": "down"})

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 7,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="801")}"},
    )

    assert response.status_code == 200, response.text
    rows = db_session.execute(
        text("SELECT id, inputs_hash, plan FROM meal_plans WHERE user_id = '801'")
    ).fetchall()
    assert len(rows) == 1
    assert rows[0].inputs_hash is not None
    assert rows[0].plan["fallback"] is True


# Validation des inputs et de l'authentification


def test_missing_authorization_returns_401(client: TestClient) -> None:
    response = client.post(
        "/api/v1/generate-meal-plan",
        json={"diet_type": "omnivore"},
    )
    assert response.status_code == 401


def test_invalid_jwt_returns_401(client: TestClient) -> None:
    response = client.post(
        "/api/v1/generate-meal-plan",
        json={"diet_type": "omnivore"},
        headers={"Authorization": "Bearer not-a-real-jwt"},
    )
    assert response.status_code == 401


def test_jwt_with_non_numeric_sub_is_accepted(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    # Depuis le refactor user_id en str (V14 + nanoID better-auth), un sub
    # non numerique est un cas nominal. On verifie que le router ne casse
    # pas avant le LLM (sans environnement Mistral/Ollama dispo en test,
    # on accepte 5xx au-dela de l'auth ; pas de 401).
    token = valid_jwt(user_id="uuid-not-a-number")
    response = client.post(
        "/api/v1/generate-meal-plan",
        json={"diet_type": "omnivore"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code != 401


def test_invalid_diet_type_returns_422(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    response = client.post(
        "/api/v1/generate-meal-plan",
        json={"diet_type": "carnivore", "duration_days": 1},
        headers={"Authorization": f"Bearer {valid_jwt(user_id="900")}"},
    )
    assert response.status_code == 422


def test_invalid_health_goal_returns_422(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    response = client.post(
        "/api/v1/generate-meal-plan",
        json={"diet_type": "omnivore", "health_goal": "marathon"},
        headers={"Authorization": f"Bearer {valid_jwt(user_id="901")}"},
    )
    assert response.status_code == 422


def test_negative_duration_returns_422(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    response = client.post(
        "/api/v1/generate-meal-plan",
        json={"diet_type": "omnivore", "duration_days": 0},
        headers={"Authorization": f"Bearer {valid_jwt(user_id="902")}"},
    )
    assert response.status_code == 422


def test_missing_diet_type_returns_422(
    client: TestClient,
    valid_jwt: Callable[..., str],
) -> None:
    response = client.post(
        "/api/v1/generate-meal-plan",
        json={"duration_days": 7},
        headers={"Authorization": f"Bearer {valid_jwt(user_id="903")}"},
    )
    assert response.status_code == 422


# Slice 7 PRD #45 : compliance_status + compliance_warnings + 503 infaisable.


def test_response_includes_compliance_status_and_warnings_on_full(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
) -> None:
    """Plan LLM 100% conforme : compliance_status='full', warnings vides."""
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="1000")}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["compliance_status"] == "full"
    assert body["compliance_warnings"] == []


def test_response_includes_static_fallback_when_ollama_down(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
) -> None:
    """Ollama injoignable apres tous les retries : status=static_fallback + warning."""
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        503, json={"error": "down"}
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 7,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="1001")}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["compliance_status"] == "static_fallback"
    assert body["compliance_warnings"]
    assert any("statique" in w.lower() for w in body["compliance_warnings"])


def test_infeasible_constraints_returns_503_with_explicit_body(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allergie + plan statique allergene : DeCRIM-light leve, router renvoie 503."""
    bad_meal = {
        "name": "Pad thai",
        "macros": {
            "calories": 500,
            "protein_g": 25.0,
            "carbs_g": 50.0,
            "fat_g": 15.0,
        },
        "ingredients": ["sauce aux arachides", "nouilles"],
        "est_budget_eur": 5.0,
        "prep_time_min": 15,
    }
    bad_plan = {
        "fallback": False,
        "days": [{"day": 1, "meals": [bad_meal]}],
    }
    bad_meal_resp = {"response": json.dumps(bad_meal), "done": True}
    # Sequence DeCRIM-light : initial allergene + 2 retries partiels allergenes
    # + retry plan complet (garde-fou) allergene -> fallback statique allergene -> 503.
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_payload(bad_plan)),
            httpx.Response(200, json=bad_meal_resp),
            httpx.Response(200, json=bad_meal_resp),
            httpx.Response(200, json=_ollama_payload(bad_plan)),
        ]
    )
    # On force le plan statique a etre allergene aussi : la combinaison est
    # vraiment infaisable.
    monkeypatch.setattr(
        "app.services.decrim_retry_orchestrator.load_fallback_plan",
        lambda _goal, _diet: bad_plan,
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": ["arachides"],
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="1002")}"},
    )

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["infeasible"] is True
    assert "infaisables" in body["detail"].lower()
    assert "allergies" in body["detail"].lower()


def test_compliance_persisted_to_meal_plans_table(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    db_session: Session,
) -> None:
    """compliance_status / compliance_warnings sont ecrits dans meal_plans."""
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="1003")}"},
    )

    assert response.status_code == 200, response.text
    row = db_session.execute(
        text(
            "SELECT compliance_status, compliance_warnings "
            "FROM meal_plans WHERE user_id = '1003'"
        )
    ).fetchone()
    assert row is not None
    assert row.compliance_status == "full"
    # _persist_plan envoie systematiquement une liste (eventuellement vide)
    # cote SQLAlchemy : Postgres stocke un ARRAY vide, jamais NULL.
    assert row.compliance_warnings == []


# Fallback chain inter-providers (issue #73 / PRD #71 slice 2).


# Slice 3 PRD #71 : selecteur user du backend LLM via /me/preferences.


def test_user_pref_ollama_persists_llm_backend_used_ollama(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    db_session: Session,
) -> None:
    """User pref preferred_llm=ollama -> plan genere via Ollama et trace en BDD."""
    headers = {"Authorization": f"Bearer {valid_jwt(user_id="1100")}"}
    client.patch(
        "/api/v1/me/preferences", json={"preferred_llm": "ollama"}, headers=headers
    )
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers=headers,
    )

    assert response.status_code == 200, response.text
    row = db_session.execute(
        text("SELECT llm_backend_used FROM meal_plans WHERE user_id = '1100'")
    ).fetchone()
    assert row is not None
    assert row.llm_backend_used == "ollama"


def test_user_pref_switch_invalidates_cache_and_regenerates(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """Switch pref Ollama -> Mistral : meme inputs produisent un nouveau plan.

    Sans le filtre cache backend-aware (AC slice 3), le 2eme appel servirait
    le plan cache Ollama. Verifie la separation effective du cache par backend.
    """
    monkeypatch.setattr(settings, "mistral_api_key", "sk-test-valid")
    headers = {"Authorization": f"Bearer {valid_jwt(user_id="1101")}"}
    body = {
        "health_goal": "balance",
        "diet_type": "omnivore",
        "duration_days": 1,
        "allergies": [],
        "budget_eur_per_day": 15,
    }

    client.patch(
        "/api/v1/me/preferences", json={"preferred_llm": "ollama"}, headers=headers
    )
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan(marker_calories=111))
    )
    mock_ollama.post("https://api.mistral.ai/v1/chat/completions").respond(
        200,
        json={
            "choices": [
                {"message": {"content": json.dumps(_valid_plan(marker_calories=222))}}
            ]
        },
    )

    first = client.post("/api/v1/generate-meal-plan", json=body, headers=headers)
    assert first.status_code == 200, first.text

    client.patch(
        "/api/v1/me/preferences", json={"preferred_llm": "mistral"}, headers=headers
    )

    second = client.post("/api/v1/generate-meal-plan", json=body, headers=headers)
    assert second.status_code == 200, second.text

    # 2 lignes meal_plans : une par backend, hash identique mais backend distinct.
    rows = db_session.execute(
        text(
            "SELECT llm_backend_used FROM meal_plans WHERE user_id = '1101' "
            "ORDER BY generated_at"
        )
    ).fetchall()
    assert [r.llm_backend_used for r in rows] == ["ollama", "mistral"]
    # Le 2eme appel renvoie le plan Mistral (marker 222), pas le cache Ollama (111).
    assert second.json()["days"][0]["meals"][0]["macros"]["calories"] == 222


def test_generate_meal_plan_falls_back_to_ollama_when_mistral_unavailable(
    client: TestClient,
    valid_jwt: Callable[..., str],
    mock_ollama: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
    db_session: Session,
) -> None:
    """Demo soutenance : Mistral KO -> bascule Ollama -> warning explicite.

    Repond a l'exigence PDF MSPR2 III.3 (fallback inter-API externes). Verifie
    aussi que le compliance_warning est persiste dans `meal_plans` pour audit.
    """
    monkeypatch.setattr(settings, "default_llm", "mistral")
    monkeypatch.setattr(settings, "mistral_api_key", "sk-test-invalid")

    mock_ollama.post("https://api.mistral.ai/v1/chat/completions").respond(
        401, json={"error": "unauthorized"}
    )
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_payload(_valid_plan())
    )

    response = client.post(
        "/api/v1/generate-meal-plan",
        json={
            "health_goal": "balance",
            "diet_type": "omnivore",
            "duration_days": 1,
            "allergies": [],
            "budget_eur_per_day": 15,
        },
        headers={"Authorization": f"Bearer {valid_jwt(user_id="1004")}"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["fallback"] is False
    assert body["compliance_status"] == "full"
    assert any(
        "Mistral indisponible" in w and "Ollama" in w
        for w in body["compliance_warnings"]
    ), f"compliance_warnings={body['compliance_warnings']!r}"

    row = db_session.execute(
        text("SELECT compliance_warnings FROM meal_plans WHERE user_id = '1004'")
    ).fetchone()
    assert row is not None
    assert any("Mistral indisponible" in w for w in row.compliance_warnings)
