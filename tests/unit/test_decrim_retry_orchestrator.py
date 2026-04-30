from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest
import respx

from app.models.schemas import FallbackMealPlan, PlanInputs
from app.services.decrim_retry_orchestrator import (
    ComplianceStatus,
    InfeasibleConstraintsError,
    generate_with_retry,
)


# Helpers : reponses Ollama deterministe pour les tests.


def _ollama_response(payload: dict[str, Any] | str) -> dict[str, Any]:
    body = payload if isinstance(payload, str) else json.dumps(payload)
    return {"response": body, "done": True}


def _meal(
    name: str = "Salade poulet",
    ingredients: list[str] | None = None,
    cost: float = 4.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "macros": {
            "calories": 500,
            "protein_g": 30.0,
            "carbs_g": 40.0,
            "fat_g": 18.0,
        },
        "ingredients": ingredients or ["poulet", "salade", "tomate"],
        "est_budget_eur": cost,
        "prep_time_min": 15,
    }


def _plan_dict(
    meals_per_day: list[list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Plan complet conforme au schema FallbackMealPlan."""
    if meals_per_day is None:
        meals_per_day = [[_meal()]]
    return {
        "fallback": False,
        "days": [
            {"day": idx + 1, "meals": meals} for idx, meals in enumerate(meals_per_day)
        ],
    }


# T1 : tracer bullet. Plan valide au 1er essai -> ComplianceStatus.full.


@pytest.mark.asyncio
async def test_success_first_try_returns_full_status(
    mock_ollama: respx.MockRouter,
) -> None:
    inputs = PlanInputs(user_id=1, objective="balance", duration_days=1)
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(_plan_dict())
    )

    plan, status = await generate_with_retry(inputs)

    assert isinstance(plan, FallbackMealPlan)
    assert status is ComplianceStatus.full
    assert plan.days[0].meals[0].name == "Salade poulet"


# T2 : retry partiel sur allergie. Le repas violant est regenere uniquement.


@pytest.mark.asyncio
async def test_partial_retry_on_allergy_succeeds_at_attempt_two(
    mock_ollama: respx.MockRouter,
) -> None:
    inputs = PlanInputs(
        user_id=2,
        objective="balance",
        duration_days=1,
        allergies=["arachides"],
    )
    bad_meal = _meal(
        name="Pad thai", ingredients=["sauce aux arachides", "nouilles"], cost=4.0
    )
    safe_meal = _meal(name="Salade poulet", ingredients=["poulet", "salade"], cost=4.0)
    bad_plan = _plan_dict([[bad_meal]])
    # 2eme reponse : regeneration ciblee de l'unique repas (forme : Meal seul).
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(bad_plan)),
            httpx.Response(200, json=_ollama_response(safe_meal)),
        ]
    )

    plan, status = await generate_with_retry(inputs)

    assert status is ComplianceStatus.full
    assert plan.days[0].meals[0].name == "Salade poulet"
    assert "arachides" not in " ".join(plan.days[0].meals[0].ingredients).lower()


# T3 : retry complet du jour qui depasse le budget.


@pytest.mark.asyncio
async def test_full_day_retry_on_budget_succeeds_at_attempt_two(
    mock_ollama: respx.MockRouter,
) -> None:
    inputs = PlanInputs(
        user_id=3,
        objective="weight_loss",
        duration_days=2,
        budget_per_day=10.0,
    )
    # Plan initial : jour 1 dans le budget, jour 2 trop cher (3 repas a 5 EUR = 15).
    expensive_day_meals = [_meal(name=f"Plat luxe {i}", cost=5.0) for i in range(3)]
    cheap_day_meals = [_meal(name="Plat simple", cost=3.0)]
    bad_plan = _plan_dict([cheap_day_meals, expensive_day_meals])
    # Jour 2 regenere : meme nombre de repas mais sous budget (3 * 3 = 9 < 10).
    cheaper_day = {
        "day": 2,
        "meals": [_meal(name=f"Plat economique {i}", cost=3.0) for i in range(3)],
    }
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(bad_plan)),
            httpx.Response(200, json=_ollama_response(cheaper_day)),
        ]
    )

    plan, status = await generate_with_retry(inputs)

    assert status is ComplianceStatus.full
    day_2 = next(d for d in plan.days if d.day == 2)
    assert sum(m.est_budget_eur for m in day_2.meals) <= 10.0
    assert all("economique" in m.name for m in day_2.meals)


# T4 : 3 retries echoues + fallback statique allergene -> InfeasibleConstraintsError.


@pytest.mark.asyncio
async def test_three_retries_fail_then_fallback_with_allergy_raises(
    mock_ollama: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = PlanInputs(
        user_id=4,
        objective="balance",
        duration_days=1,
        allergies=["arachides"],
    )
    bad_meal = _meal(name="Pad thai", ingredients=["sauce aux arachides", "riz"])
    bad_plan = _plan_dict([[bad_meal]])
    # Sequence : plan initial allergene -> 2 retries partiels allergenes -> garde-fou
    # bascule en retry plan complet (encore allergene) -> fallback statique allergene
    # -> InfeasibleConstraintsError.
    bad_meal_resp = _ollama_response(bad_meal)
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(bad_plan)),
            httpx.Response(200, json=bad_meal_resp),
            httpx.Response(200, json=bad_meal_resp),
            httpx.Response(200, json=_ollama_response(bad_plan)),
        ]
    )

    # Plan statique de fallback aussi allergene -> 503.
    monkeypatch.setattr(
        "app.services.decrim_retry_orchestrator.load_fallback_plan",
        lambda _goal, _diet: bad_plan,
    )

    with pytest.raises(InfeasibleConstraintsError):
        await generate_with_retry(inputs)


# T5 : 3 retries echoues sur le budget -> ComplianceStatus.partial_budget.


@pytest.mark.asyncio
async def test_three_retries_fail_on_budget_returns_partial_budget(
    mock_ollama: respx.MockRouter,
) -> None:
    inputs = PlanInputs(
        user_id=5,
        objective="balance",
        duration_days=1,
        budget_per_day=10.0,
    )
    expensive_meals = [_meal(name=f"Plat {i}", cost=5.0) for i in range(3)]
    expensive_plan = _plan_dict([expensive_meals])
    expensive_day = {"day": 1, "meals": expensive_meals}
    # Toutes les regenerations renvoient encore au-dessus du budget : 3 * 5 = 15 > 10.
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(expensive_plan)),
            httpx.Response(200, json=_ollama_response(expensive_day)),
            httpx.Response(200, json=_ollama_response(expensive_day)),
            httpx.Response(200, json=_ollama_response(expensive_day)),
        ]
    )

    plan, status = await generate_with_retry(inputs)

    assert status is ComplianceStatus.partial_budget
    # Le plan retourne reste celui issu du LLM (pas de fallback statique sur budget seul).
    assert sum(m.est_budget_eur for m in plan.days[0].meals) == 15.0


# T6 : garde-fou anti-cycle. Apres 2 retries partiels sur le meme repas,
# basculer en retry complet du plan.


@pytest.mark.asyncio
async def test_anti_cycle_switches_to_full_plan_retry(
    mock_ollama: respx.MockRouter,
) -> None:
    inputs = PlanInputs(
        user_id=6,
        objective="balance",
        duration_days=1,
        allergies=["arachides"],
    )
    bad_meal = _meal(name="Pad thai", ingredients=["sauce aux arachides", "riz"])
    safe_meal = _meal(name="Salade poulet", ingredients=["poulet", "salade"])
    bad_plan = _plan_dict([[bad_meal]])
    safe_plan = _plan_dict([[safe_meal]])

    # Ordre attendu :
    # 1) plan initial (allergie)
    # 2) retry partiel #1 (allergie)
    # 3) retry partiel #2 (allergie) -> garde-fou : count=2
    # 4) retry complet du plan (3eme retry) -> propre
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(bad_plan)),
            httpx.Response(200, json=_ollama_response(bad_meal)),
            httpx.Response(200, json=_ollama_response(bad_meal)),
            httpx.Response(200, json=_ollama_response(safe_plan)),
        ]
    )

    plan, status = await generate_with_retry(inputs)

    assert status is ComplianceStatus.full
    assert plan.days[0].meals[0].name == "Salade poulet"
