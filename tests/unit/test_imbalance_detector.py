from __future__ import annotations

from decimal import Decimal

import pytest

from app.db.models import NutritionGoal
from app.models.schemas import Imbalance
from app.services.imbalance_detector import detect_imbalances


def _goal(**kwargs) -> NutritionGoal:
    """Construit un NutritionGoal sans BDD (objet detache, pour test pur)."""
    return NutritionGoal(user_id=1, **kwargs)


def test_no_goal_returns_empty_list() -> None:
    assert detect_imbalances({"calories": 1500, "protein_g": 50}, None) == []


def test_aligned_meal_returns_empty_list() -> None:
    goal = _goal(
        calories_target=2000,
        protein_g=Decimal("100"),
        carbs_g=Decimal("250"),
        fat_g=Decimal("70"),
    )
    macros = {"calories": 600, "protein_g": 30, "carbs_g": 70, "fat_g": 20}
    assert detect_imbalances(macros, goal) == []


def test_high_calories_imbalance_yields_calories_high_kind() -> None:
    goal = _goal(calories_target=2000)
    macros = {"calories": 1500}  # 75 % du target, > 60 %

    issues = detect_imbalances(macros, goal)

    assert len(issues) == 1
    kind, message = issues[0]
    assert kind == Imbalance.calories_high
    assert "1500" in message
    assert "2000" in message


def test_low_protein_imbalance_yields_protein_low_kind() -> None:
    goal = _goal(protein_g=Decimal("100"))
    macros = {"protein_g": 10}  # < 20 % du target

    issues = detect_imbalances(macros, goal)

    assert len(issues) == 1
    kind, _ = issues[0]
    assert kind == Imbalance.protein_low


def test_high_carbs_imbalance_yields_carbs_high_kind() -> None:
    goal = _goal(carbs_g=Decimal("200"))
    macros = {"carbs_g": 180}  # > 70 % du target

    issues = detect_imbalances(macros, goal)

    assert [kind for kind, _ in issues] == [Imbalance.carbs_high]


def test_high_fat_imbalance_yields_fat_high_kind() -> None:
    goal = _goal(fat_g=Decimal("60"))
    macros = {"fat_g": 50}  # > 70 % du target

    issues = detect_imbalances(macros, goal)

    assert [kind for kind, _ in issues] == [Imbalance.fat_high]


def test_imbalance_skips_unset_macros() -> None:
    goal = _goal(calories_target=2000, protein_g=Decimal("100"))
    macros = {"calories": 1500}  # protein_g absent

    issues = detect_imbalances(macros, goal)

    assert [kind for kind, _ in issues] == [Imbalance.calories_high]


def test_imbalance_returns_multiple_kinds_in_stable_order() -> None:
    goal = _goal(
        calories_target=2000,
        protein_g=Decimal("100"),
        carbs_g=Decimal("200"),
        fat_g=Decimal("60"),
    )
    macros = {"calories": 1500, "protein_g": 5, "carbs_g": 180, "fat_g": 50}

    issues = detect_imbalances(macros, goal)
    kinds = [kind for kind, _ in issues]

    # Ordre stable : calories, protein, carbs, fat (utile pour hash de cache).
    assert kinds == [
        Imbalance.calories_high,
        Imbalance.protein_low,
        Imbalance.carbs_high,
        Imbalance.fat_high,
    ]


@pytest.mark.parametrize(
    "ratio,expected",
    [
        (0.55, []),  # 55 % du target : pas d'alerte
        (0.61, [Imbalance.calories_high]),  # > 60 % : alerte
    ],
)
def test_calories_threshold_at_60_percent(ratio: float, expected: list[Imbalance]) -> None:
    goal = _goal(calories_target=2000)
    macros = {"calories": int(2000 * ratio)}
    assert [kind for kind, _ in detect_imbalances(macros, goal)] == expected
