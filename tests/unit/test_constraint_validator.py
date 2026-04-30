from __future__ import annotations

from app.models.schemas import FallbackMealPlan
from app.services.constraint_validator import (
    ConstraintSpec,
    ViolationType,
    validate,
)


def _plan(
    *,
    ingredients: list[str] | None = None,
    cost_eur: float = 5.0,
    day: int = 1,
) -> FallbackMealPlan:
    return FallbackMealPlan.model_validate(
        {
            "fallback": False,
            "days": [
                {
                    "day": day,
                    "meals": [
                        {
                            "name": "lunch",
                            "macros": {
                                "calories": 500,
                                "protein_g": 30,
                                "carbs_g": 60,
                                "fat_g": 15,
                            },
                            "ingredients": list(ingredients or ["riz", "poulet"]),
                            "est_budget_eur": cost_eur,
                            "prep_time_min": 20,
                        }
                    ],
                }
            ],
        }
    )


def test_validate_returns_empty_list_when_spec_has_no_constraints() -> None:
    plan = _plan(ingredients=["riz", "poulet"], cost_eur=4.0)
    spec = ConstraintSpec()  # pas d'allergies, pas de budget, pas de regime

    assert validate(plan, spec) == []


def test_allergy_matched_at_word_boundary() -> None:
    plan = _plan(ingredients=["lait ecreme", "cafe"])
    spec = ConstraintSpec(allergies=["lait"])

    violations = validate(plan, spec)

    assert len(violations) == 1
    v = violations[0]
    assert v.type is ViolationType.allergy
    assert v.day == 1
    assert v.meal_index == 0
    assert v.ingredient_or_amount == "lait ecreme"
    assert "lait" in v.message


def test_allergy_does_not_false_positive_on_substring() -> None:
    # 'lait' ne doit pas matcher 'laitue', 'oeuf' ne doit pas matcher 'boeuf'.
    plan = _plan(ingredients=["laitue", "boeuf braise", "tomate"])
    spec = ConstraintSpec(allergies=["lait", "oeuf"])

    assert validate(plan, spec) == []


def test_allergy_match_strips_accents_and_case() -> None:
    plan = _plan(ingredients=["Lait écrémé", "céréales"])
    spec = ConstraintSpec(allergies=["lait"])

    violations = validate(plan, spec)

    assert len(violations) == 1
    assert violations[0].ingredient_or_amount == "Lait écrémé"


def test_budget_within_limit_returns_no_violation() -> None:
    plan = _plan(cost_eur=8.0)
    spec = ConstraintSpec(max_daily_budget_eur=10.0)

    assert validate(plan, spec) == []


def test_budget_exceeded_reports_day_total() -> None:
    plan = _plan(cost_eur=12.0, day=2)
    spec = ConstraintSpec(max_daily_budget_eur=10.0)

    violations = validate(plan, spec)

    assert len(violations) == 1
    v = violations[0]
    assert v.type is ViolationType.budget
    assert v.day == 2
    assert v.meal_index is None
    assert v.ingredient_or_amount == 12.0
    assert "10" in v.message  # mentionne la limite


def test_diet_vegan_flags_meat_ingredient() -> None:
    plan = _plan(ingredients=["poulet roti", "riz"])
    spec = ConstraintSpec(diet_type="vegan")

    violations = validate(plan, spec)

    diet_violations = [v for v in violations if v.type is ViolationType.diet]
    assert len(diet_violations) == 1
    assert diet_violations[0].ingredient_or_amount == "poulet roti"
    assert diet_violations[0].day == 1
    assert diet_violations[0].meal_index == 0


def test_diet_sans_gluten_flags_ble() -> None:
    plan = _plan(ingredients=["pates au ble", "tomate"])
    spec = ConstraintSpec(diet_type="sans_gluten")

    violations = validate(plan, spec)

    diet_violations = [v for v in violations if v.type is ViolationType.diet]
    # 'pates' et 'ble' sont tous deux bannis -> deux violations sur le meme ingredient.
    assert len(diet_violations) >= 1
    assert all(v.day == 1 and v.meal_index == 0 for v in diet_violations)


def test_diet_omnivore_has_no_banned_set() -> None:
    plan = _plan(ingredients=["poulet", "boeuf", "riz"])
    spec = ConstraintSpec(diet_type="omnivore")

    assert validate(plan, spec) == []


def _multi_day_plan(days: list[dict]) -> FallbackMealPlan:
    return FallbackMealPlan.model_validate({"fallback": False, "days": days})


def _meal(*, ingredients: list[str], cost: float = 5.0) -> dict:
    return {
        "name": "meal",
        "macros": {"calories": 500, "protein_g": 30, "carbs_g": 60, "fat_g": 15},
        "ingredients": ingredients,
        "est_budget_eur": cost,
        "prep_time_min": 20,
    }


def test_multi_violations_are_all_reported_no_short_circuit() -> None:
    plan = _multi_day_plan(
        [
            {
                "day": 1,
                "meals": [
                    _meal(ingredients=["poulet roti", "lait"], cost=8.0),
                    _meal(ingredients=["riz"], cost=5.0),  # day 1 total = 13 EUR
                ],
            },
            {
                "day": 2,
                "meals": [_meal(ingredients=["tofu", "riz"], cost=3.0)],
            },
        ]
    )
    spec = ConstraintSpec(
        allergies=["lait"], max_daily_budget_eur=10.0, diet_type="vegan"
    )

    violations = validate(plan, spec)

    by_type = {t: [v for v in violations if v.type is t] for t in ViolationType}
    # Allergie 'lait' dans le repas 0 du jour 1
    assert any(
        v.day == 1 and v.meal_index == 0 and v.ingredient_or_amount == "lait"
        for v in by_type[ViolationType.allergy]
    )
    # Diet vegan : 'poulet roti' dans le repas 0 du jour 1, et 'lait' aussi
    assert any(
        v.day == 1 and v.meal_index == 0 and v.ingredient_or_amount == "poulet roti"
        for v in by_type[ViolationType.diet]
    )
    # Budget jour 1 = 13 EUR > 10 EUR
    assert any(
        v.day == 1 and v.meal_index is None and v.ingredient_or_amount == 13.0
        for v in by_type[ViolationType.budget]
    )
    # Le jour 2 est conforme : pas de violation jour 2
    assert all(v.day != 2 for v in violations)


def test_plan_respecting_all_three_constraints_returns_empty() -> None:
    plan = _multi_day_plan(
        [
            {
                "day": 1,
                "meals": [
                    _meal(ingredients=["tofu", "quinoa", "epinards"], cost=6.0),
                    _meal(ingredients=["lentilles", "carottes"], cost=3.5),
                ],
            }
        ]
    )
    spec = ConstraintSpec(
        allergies=["arachide"], max_daily_budget_eur=12.0, diet_type="vegan"
    )

    assert validate(plan, spec) == []
