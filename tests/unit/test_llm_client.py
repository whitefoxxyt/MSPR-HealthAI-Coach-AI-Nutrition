from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import text

from app.models.schemas import (
    FallbackMealPlan,
    PlanInputs,
)
from app.services.llm_client import (
    compute_inputs_hash,
    generate_plan,
)


def _ollama_response(plan_json: dict[str, Any]) -> dict[str, Any]:
    """Forme une reponse Ollama /api/generate enveloppant un JSON metier."""
    return {"response": json.dumps(plan_json), "done": True}


def _valid_plan_dict(marker_calories: int = 1800) -> dict[str, Any]:
    """Plan minimal au schema FallbackMealPlan. marker_calories permet de
    distinguer un plan d'un autre dans les tests de cache."""
    return {
        "fallback": False,
        "days": [
            {
                "day": 1,
                "meals": [
                    {
                        "name": "Salade poulet",
                        "macros": {
                            "calories": marker_calories,
                            "protein_g": 35.0,
                            "carbs_g": 20.0,
                            "fat_g": 18.0,
                        },
                        "ingredients": ["poulet", "salade", "tomate"],
                        "est_budget_eur": 4.5,
                        "prep_time_min": 15,
                    },
                ],
            },
        ],
    }


def _meal_calories(plan: FallbackMealPlan) -> int:
    return plan.days[0].meals[0].macros.calories


# Le hash doit etre stable face a l'ordre des allergies et des cles JSON.
def test_inputs_hash_canonicalizes_allergies_order() -> None:
    a = PlanInputs(
        user_id=1,
        objective="weight_loss",
        duration_days=7,
        allergies=["gluten", "arachides", "lactose"],
        diet_type="balance",
    )
    b = PlanInputs(
        user_id=1,
        objective="weight_loss",
        duration_days=7,
        allergies=["lactose", "gluten", "arachides"],
        diet_type="balance",
    )
    assert compute_inputs_hash(a) == compute_inputs_hash(b)


def test_inputs_hash_distinguishes_user_id() -> None:
    a = PlanInputs(user_id=1, objective="muscle_gain", duration_days=7)
    b = PlanInputs(user_id=2, objective="muscle_gain", duration_days=7)
    assert compute_inputs_hash(a) != compute_inputs_hash(b)


def test_inputs_hash_distinguishes_objective() -> None:
    a = PlanInputs(user_id=1, objective="weight_loss", duration_days=7)
    b = PlanInputs(user_id=1, objective="muscle_gain", duration_days=7)
    assert compute_inputs_hash(a) != compute_inputs_hash(b)


def test_inputs_hash_returns_sha256_hex_string() -> None:
    inputs = PlanInputs(user_id=1, objective="balance", duration_days=7)
    h = compute_inputs_hash(inputs)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# generate_plan : chemin nominal


@pytest.mark.asyncio
async def test_generate_plan_success_first_try(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=42, objective="balance", duration_days=7)
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(_valid_plan_dict())
    )

    plan = await generate_plan(inputs, db_session)

    assert isinstance(plan, FallbackMealPlan)
    assert plan.fallback is False
    assert _meal_calories(plan) == 1800
    assert plan.days[0].meals[0].name == "Salade poulet"


@pytest.mark.asyncio
async def test_generate_plan_persists_with_inputs_hash(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=42, objective="weight_loss", duration_days=3)
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(_valid_plan_dict())
    )

    await generate_plan(inputs, db_session)

    row = db_session.execute(
        text("SELECT user_id, inputs_hash, plan FROM meal_plans WHERE user_id = :uid"),
        {"uid": 42},
    ).fetchone()
    assert row is not None
    assert row.inputs_hash == compute_inputs_hash(inputs)
    assert row.user_id == 42
    assert row.plan["days"][0]["meals"][0]["macros"]["calories"] == 1800


@pytest.mark.asyncio
async def test_generate_plan_calls_ollama_with_json_format(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=1, objective="balance", duration_days=7)
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(_valid_plan_dict())
    )

    await generate_plan(inputs, db_session)

    assert route.called
    body = json.loads(route.calls.last.request.content)
    # On envoie le schema Pydantic en parametre format pour forcer du JSON valide.
    assert "format" in body
    assert isinstance(body["format"], dict)
    assert body["format"].get("type") == "object"
    assert "stream" in body and body["stream"] is False


# generate_plan : retry


@pytest.mark.asyncio
async def test_generate_plan_retries_after_first_failure(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=10, objective="balance", duration_days=2)
    responses = [
        httpx.Response(500, json={"error": "boom"}),
        httpx.Response(200, json=_ollama_response(_valid_plan_dict())),
    ]
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(side_effect=responses)

    plan = await generate_plan(inputs, db_session)

    assert plan.fallback is False
    assert _meal_calories(plan) == 1800


@pytest.mark.asyncio
async def test_generate_plan_retries_on_invalid_json(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=11, objective="balance", duration_days=2)
    responses = [
        httpx.Response(200, json={"response": "pas du JSON valide", "done": True}),
        httpx.Response(200, json=_ollama_response(_valid_plan_dict())),
    ]
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(side_effect=responses)

    plan = await generate_plan(inputs, db_session)

    assert plan.fallback is False
    assert plan.days[0].day == 1


# generate_plan : fallback


@pytest.mark.asyncio
async def test_generate_plan_falls_back_after_max_attempts(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=20, objective="weight_loss", duration_days=1)
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        500, json={"error": "boom"}
    )

    fallback_calls: list[tuple[str, str]] = []

    def loader(objective: str, diet_type: str) -> dict:
        fallback_calls.append((objective, diet_type))
        return {
            "fallback": True,
            "days": [
                {
                    "day": 1,
                    "meals": [
                        {
                            "name": "Plan de secours",
                            "macros": {
                                "calories": 600,
                                "protein_g": 20.0,
                                "carbs_g": 80.0,
                                "fat_g": 15.0,
                            },
                            "ingredients": ["riz", "legumes"],
                            "est_budget_eur": 2.0,
                            "prep_time_min": 10,
                        }
                    ],
                }
            ],
        }

    plan = await generate_plan(inputs, db_session, fallback_loader=loader)

    assert plan.fallback is True
    assert _meal_calories(plan) == 600
    assert plan.days[0].meals[0].name == "Plan de secours"
    assert fallback_calls == [("weight_loss", "")]


@pytest.mark.asyncio
async def test_generate_plan_makes_exactly_three_attempts_before_fallback(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=21, objective="balance", duration_days=1)
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        500, json={"error": "boom"}
    )

    plan = await generate_plan(
        inputs, db_session, fallback_loader=lambda *_: {"fallback": True, "days": []}
    )

    assert plan.fallback is True
    assert route.call_count == 3  # 1 initial + 2 retries


@pytest.mark.asyncio
async def test_generate_plan_fallback_without_loader_returns_empty_plan(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=22, objective="balance", duration_days=1)
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        500, json={"error": "boom"}
    )

    plan = await generate_plan(inputs, db_session)

    assert plan.fallback is True
    assert plan.days == []


# generate_plan : cache


@pytest.mark.asyncio
async def test_generate_plan_returns_cached_without_calling_ollama(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=30, objective="balance", duration_days=2)
    cached = _valid_plan_dict(marker_calories=9999)  # marqueur pour distinguer du mock
    db_session.execute(
        text(
            "INSERT INTO meal_plans (user_id, plan, objective, inputs_hash, generated_at) "
            "VALUES (:uid, CAST(:plan AS JSONB), :obj, :h, NOW())"
        ),
        {
            "uid": 30,
            "plan": json.dumps(cached),
            "obj": "balance",
            "h": compute_inputs_hash(inputs),
        },
    )
    db_session.commit()

    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(_valid_plan_dict())
    )

    plan = await generate_plan(inputs, db_session)

    assert _meal_calories(plan) == 9999
    assert route.call_count == 0


@pytest.mark.asyncio
async def test_generate_plan_ignores_cache_older_than_seven_days(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=31, objective="balance", duration_days=2)
    stale = _valid_plan_dict(marker_calories=9999)
    db_session.execute(
        text(
            "INSERT INTO meal_plans (user_id, plan, objective, inputs_hash, generated_at) "
            "VALUES (:uid, CAST(:plan AS JSONB), :obj, :h, NOW() - INTERVAL '8 days')"
        ),
        {
            "uid": 31,
            "plan": json.dumps(stale),
            "obj": "balance",
            "h": compute_inputs_hash(inputs),
        },
    )
    db_session.commit()

    fresh = _valid_plan_dict(marker_calories=1234)
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(fresh)
    )

    plan = await generate_plan(inputs, db_session)

    assert _meal_calories(plan) == 1234


@pytest.mark.asyncio
async def test_generate_plan_bypass_cache_calls_ollama_even_with_hit(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=32, objective="balance", duration_days=2)
    cached = _valid_plan_dict(marker_calories=9999)
    db_session.execute(
        text(
            "INSERT INTO meal_plans (user_id, plan, objective, inputs_hash, generated_at) "
            "VALUES (:uid, CAST(:plan AS JSONB), :obj, :h, NOW())"
        ),
        {
            "uid": 32,
            "plan": json.dumps(cached),
            "obj": "balance",
            "h": compute_inputs_hash(inputs),
        },
    )
    db_session.commit()

    fresh = _valid_plan_dict(marker_calories=4242)
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(fresh)
    )

    plan = await generate_plan(inputs, db_session, bypass_cache=True)

    assert _meal_calories(plan) == 4242
    assert route.call_count == 1
    # Verifie qu'une nouvelle ligne a ete inseree (cache + nouveau plan).
    count = db_session.execute(
        text("SELECT COUNT(*) FROM meal_plans WHERE inputs_hash = :h"),
        {"h": compute_inputs_hash(inputs)},
    ).scalar()
    assert count == 2


# generate_plan : validation Pydantic + regles metier


@pytest.mark.asyncio
async def test_generate_plan_rejects_plan_with_allergic_ingredient(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(
        user_id=40,
        objective="balance",
        duration_days=1,
        allergies=["arachides"],
    )
    bad = _valid_plan_dict()
    # Le plan contient un ingredient interdit -> validation metier doit rejeter.
    bad["days"][0]["meals"][0]["ingredients"] = ["sauce aux arachides", "riz"]
    good = _valid_plan_dict()
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(bad)),
            httpx.Response(200, json=_ollama_response(good)),
        ]
    )

    plan = await generate_plan(inputs, db_session)

    # Le second essai (sans allergene) est accepte.
    assert plan.fallback is False
    assert all(
        "arachide" not in ing.lower()
        for day in plan.days
        for meal in day.meals
        for ing in meal.ingredients
    )


@pytest.mark.asyncio
async def test_generate_plan_does_not_false_positive_lait_in_laitue(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    """Anti-regression : 'lait' (allergie) ne doit pas matcher 'laitue' (ingredient)."""
    inputs = PlanInputs(
        user_id=43,
        objective="balance",
        duration_days=1,
        allergies=["lait"],
    )
    safe = _valid_plan_dict()
    # 'laitue' contient 'lait' en sous-chaine mais c'est un mot different.
    # 'oeuf' / 'boeuf' meme piege en francais.
    safe["days"][0]["meals"][0]["ingredients"] = ["laitue", "boeuf", "tomate"]
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json=_ollama_response(safe)
    )

    plan = await generate_plan(inputs, db_session)

    assert plan.fallback is False
    assert route.call_count == 1  # accepte du premier coup, aucun retry
    assert plan.days[0].meals[0].ingredients == ["laitue", "boeuf", "tomate"]


@pytest.mark.asyncio
async def test_generate_plan_rejects_allergen_with_accents(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    """L'allergene 'lait' doit matcher 'Lait ecreme' meme avec capitalisation/accent."""
    inputs = PlanInputs(
        user_id=44,
        objective="balance",
        duration_days=1,
        allergies=["lait"],
    )
    bad = _valid_plan_dict()
    bad["days"][0]["meals"][0]["ingredients"] = ["Lait écrémé", "céréales"]
    good = _valid_plan_dict()
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(bad)),
            httpx.Response(200, json=_ollama_response(good)),
        ]
    )

    plan = await generate_plan(inputs, db_session)

    assert plan.fallback is False
    assert all(
        "lait" not in ing.lower().replace("é", "e")
        for day in plan.days
        for meal in day.meals
        for ing in meal.ingredients
    )


@pytest.mark.asyncio
async def test_generate_plan_rejects_invalid_pydantic_structure(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    inputs = PlanInputs(user_id=41, objective="balance", duration_days=1)
    invalid = {"days": "not a list"}  # type incorrect : doit etre rejete par Pydantic
    good = _valid_plan_dict()
    mock_ollama.post(re.compile(r".*/api/generate$")).mock(
        side_effect=[
            httpx.Response(200, json=_ollama_response(invalid)),
            httpx.Response(200, json=_ollama_response(good)),
        ]
    )

    plan = await generate_plan(inputs, db_session)

    assert plan.fallback is False


# generate_plan : semaphore (au plus 2 inferences simultanees)


@pytest.mark.asyncio
async def test_generate_plan_semaphore_limits_concurrent_ollama_calls(
    db_session, mock_ollama: respx.MockRouter
) -> None:
    in_flight = 0
    max_observed = 0

    async def slow_handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_observed
        in_flight += 1
        max_observed = max(max_observed, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, json=_ollama_response(_valid_plan_dict()))

    mock_ollama.post(re.compile(r".*/api/generate$")).mock(side_effect=slow_handler)

    inputs_list = [
        PlanInputs(user_id=50 + i, objective="balance", duration_days=1)
        for i in range(4)
    ]
    plans = await asyncio.gather(*(generate_plan(i, db_session) for i in inputs_list))

    assert all(p.fallback is False for p in plans)
    assert max_observed >= 2  # parallelisme effectif
    assert max_observed <= 2  # bornage du semaphore


# Tests generate_recommendation : voir tests/unit/test_llm_recommendation_v2.py
# (signature refactoree par l'issue #51 : list[ImbalanceTag] au lieu de
# RecommendationContext, 1 seul appel Ollama au lieu de N).
