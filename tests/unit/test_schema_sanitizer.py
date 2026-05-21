from __future__ import annotations

import json
from typing import Any

from app.models.schemas import FallbackMealPlan, Meal, MealDay
from app.services.schema_sanitizer import sanitize_for_mistral_strict


def _assert_no_unresolved_refs(node: Any) -> None:
    if isinstance(node, dict):
        assert "$ref" not in node, f"$ref residuel : {node}"
        assert "$defs" not in node, f"$defs residuel : {node}"
        for v in node.values():
            _assert_no_unresolved_refs(v)
    elif isinstance(node, list):
        for v in node:
            _assert_no_unresolved_refs(v)


def _assert_objects_have_additional_properties_false(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, (
                f"objet sans additionalProperties=false : {node}"
            )
        for v in node.values():
            _assert_objects_have_additional_properties_false(v)
    elif isinstance(node, list):
        for v in node:
            _assert_objects_have_additional_properties_false(v)


def test_flat_object_schema_gets_additional_properties_false() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    result = sanitize_for_mistral_strict(schema)

    assert result["additionalProperties"] is False
    assert result["type"] == "object"
    assert result["properties"]["name"] == {"type": "string"}


def test_ref_resolves_against_defs_and_defs_is_dropped() -> None:
    schema = {
        "type": "object",
        "properties": {"meal": {"$ref": "#/$defs/Meal"}},
        "required": ["meal"],
        "$defs": {
            "Meal": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            }
        },
    }

    result = sanitize_for_mistral_strict(schema)

    assert "$defs" not in result
    meal = result["properties"]["meal"]
    assert "$ref" not in meal
    assert meal["type"] == "object"
    assert meal["properties"]["name"] == {"type": "string"}
    assert meal["additionalProperties"] is False


def test_nested_object_also_gets_additional_properties_false() -> None:
    schema = {
        "type": "object",
        "properties": {
            "macros": {
                "type": "object",
                "properties": {"calories": {"type": "integer"}},
                "required": ["calories"],
            }
        },
        "required": ["macros"],
    }

    result = sanitize_for_mistral_strict(schema)

    assert result["additionalProperties"] is False
    assert result["properties"]["macros"]["additionalProperties"] is False


def test_meal_schema_is_strict_compatible() -> None:
    result = sanitize_for_mistral_strict(Meal.model_json_schema())

    _assert_no_unresolved_refs(result)
    _assert_objects_have_additional_properties_false(result)
    # Le schema reste serialisable et expose les champs metier de Meal.
    json.dumps(result)
    assert set(result["properties"].keys()) == {
        "name",
        "macros",
        "ingredients",
        "est_budget_eur",
        "prep_time_min",
    }
    # MealMacros etait dans $defs, doit etre resolu inline.
    assert result["properties"]["macros"]["type"] == "object"
    assert "calories" in result["properties"]["macros"]["properties"]


def test_meal_day_schema_is_strict_compatible() -> None:
    result = sanitize_for_mistral_strict(MealDay.model_json_schema())

    _assert_no_unresolved_refs(result)
    _assert_objects_have_additional_properties_false(result)
    # Liste de Meal -> items doit etre un objet resolu.
    meals = result["properties"]["meals"]
    assert meals["type"] == "array"
    assert meals["items"]["type"] == "object"
    assert "name" in meals["items"]["properties"]


def test_fallback_meal_plan_schema_is_strict_compatible() -> None:
    result = sanitize_for_mistral_strict(FallbackMealPlan.model_json_schema())

    _assert_no_unresolved_refs(result)
    _assert_objects_have_additional_properties_false(result)
    # Chemin complet : plan -> days[] -> meals[] -> macros.
    days = result["properties"]["days"]
    assert days["type"] == "array"
    meals_field = days["items"]["properties"]["meals"]
    assert meals_field["items"]["properties"]["macros"]["type"] == "object"
