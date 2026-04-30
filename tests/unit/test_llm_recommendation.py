from __future__ import annotations

import re
from typing import Any

import httpx
import pytest
import respx

from app.models.schemas import (
    HealthGoal,
    ImbalanceStatus,
    ImbalanceTag,
    Nutrient,
)
from app.services.llm_client import generate_recommendation


def _tag(
    nutrient: Nutrient,
    status: ImbalanceStatus,
    delta_pct: float = 0.30,
    target: float = 100.0,
    actual: float = 130.0,
    unit: str = "g",
) -> ImbalanceTag:
    return ImbalanceTag(
        nutrient=nutrient,
        status=status,
        delta_pct=delta_pct,
        target_value=target,
        actual_value=actual,
        unit=unit,
    )


@pytest.mark.asyncio
async def test_generate_recommendation_takes_list_of_tags_and_returns_string(
    db_session,
    mock_ollama: respx.MockRouter,
) -> None:
    tags = [_tag(Nutrient.calories, ImbalanceStatus.excess)]
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200,
        json={
            "response": "Reduis la portion et ajoute des legumes verts.",
            "done": True,
        },
    )

    suggestion = await generate_recommendation(
        ctx_list=tags,
        health_goal=HealthGoal.balance,
        db=db_session,
    )

    assert isinstance(suggestion, str)
    assert "legumes" in suggestion or "portion" in suggestion


@pytest.mark.asyncio
async def test_generate_recommendation_makes_exactly_one_ollama_call_for_multiple_tags(
    db_session,
    mock_ollama: respx.MockRouter,
) -> None:
    # Issue #51 : un seul appel Ollama meme avec plusieurs imbalances.
    tags = [
        _tag(Nutrient.calories, ImbalanceStatus.excess),
        _tag(Nutrient.protein_g, ImbalanceStatus.deficit),
        _tag(Nutrient.fibers_g, ImbalanceStatus.deficit),
    ]
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json={"response": "Conseil synthetique.", "done": True}
    )

    await generate_recommendation(
        ctx_list=tags,
        health_goal=HealthGoal.balance,
        db=db_session,
    )

    assert route.call_count == 1


@pytest.mark.asyncio
async def test_generate_recommendation_prompt_includes_imbalances_and_goal(
    db_session,
    mock_ollama: respx.MockRouter,
) -> None:
    captured: list[dict[str, Any]] = []

    def _handler(request: httpx.Request) -> httpx.Response:
        captured.append({"body": httpx.Request.read(request).decode()})
        return httpx.Response(200, json={"response": "ok.", "done": True})

    mock_ollama.post(re.compile(r".*/api/generate$")).mock(side_effect=_handler)

    tags = [
        _tag(Nutrient.protein_g, ImbalanceStatus.deficit),
        _tag(Nutrient.saturated_fat_g, ImbalanceStatus.excess),
    ]

    await generate_recommendation(
        ctx_list=tags,
        health_goal=HealthGoal.muscle_gain,
        db=db_session,
    )

    assert captured, "Aucun appel Ollama enregistre"
    body = captured[0]["body"]
    assert "muscle_gain" in body
    # Le prompt doit nommer chaque imbalance pour que la synthese soit ciblee.
    assert "protein" in body or "proteine" in body.lower()
    assert "satur" in body.lower() or "ags" in body.lower()


@pytest.mark.asyncio
async def test_generate_recommendation_falls_back_to_default_when_ollama_down(
    db_session,
    mock_ollama: respx.MockRouter,
) -> None:
    tags = [_tag(Nutrient.calories, ImbalanceStatus.excess)]
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        500, json={"error": "boom"}
    )

    suggestion = await generate_recommendation(
        ctx_list=tags,
        health_goal=HealthGoal.weight_loss,
        db=db_session,
    )

    assert isinstance(suggestion, str)
    assert suggestion != ""


@pytest.mark.asyncio
async def test_generate_recommendation_uses_custom_fallback_callable(
    db_session,
    mock_ollama: respx.MockRouter,
) -> None:
    tags = [_tag(Nutrient.fat_g, ImbalanceStatus.excess)]
    mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        503, json={"error": "down"}
    )

    captured_calls: list[tuple[tuple[ImbalanceTag, ...], HealthGoal]] = []

    def fb(tag_list: list[ImbalanceTag], goal: HealthGoal) -> str:
        captured_calls.append((tuple(tag_list), goal))
        return "Plan de repli."

    suggestion = await generate_recommendation(
        ctx_list=tags,
        health_goal=HealthGoal.balance,
        db=db_session,
        fallback=fb,
    )

    assert suggestion == "Plan de repli."
    assert len(captured_calls) == 1
    assert captured_calls[0][1] is HealthGoal.balance
    assert captured_calls[0][0][0].nutrient is Nutrient.fat_g


@pytest.mark.asyncio
async def test_generate_recommendation_returns_default_when_ctx_list_empty(
    db_session,
    mock_ollama: respx.MockRouter,
) -> None:
    # Pas d'imbalance -> pas d'appel LLM, on retourne une chaine vide ou neutre.
    route = mock_ollama.post(re.compile(r".*/api/generate$")).respond(
        200, json={"response": "Continue.", "done": True}
    )

    suggestion = await generate_recommendation(
        ctx_list=[],
        health_goal=HealthGoal.balance,
        db=db_session,
    )

    assert isinstance(suggestion, str)
    assert route.call_count == 0
