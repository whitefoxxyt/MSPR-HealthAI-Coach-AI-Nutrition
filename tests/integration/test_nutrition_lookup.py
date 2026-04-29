from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.nutrition_lookup import lookup_nutrition


def _insert(
    db: Session,
    *,
    food_name: str,
    calories: float = 100.0,
    protein_g: float = 10.0,
    carbs_g: float = 20.0,
    fat_g: float = 5.0,
    fiber_g: float = 2.0,
) -> None:
    db.execute(
        text(
            "INSERT INTO nutrition_entries "
            "(food_name, calories, protein_g, carbs_g, fat_g, fiber_g, source) "
            "VALUES (:food_name, :calories, :protein_g, :carbs_g, :fat_g, :fiber_g, 'TEST')"
        ),
        {
            "food_name": food_name,
            "calories": calories,
            "protein_g": protein_g,
            "carbs_g": carbs_g,
            "fat_g": fat_g,
            "fiber_g": fiber_g,
        },
    )
    db.commit()


def test_exact_match_returns_food(db_session: Session) -> None:
    _insert(
        db_session,
        food_name="Grilled Salmon",
        calories=412.0,
        protein_g=40.0,
        carbs_g=0.0,
        fat_g=27.0,
        fiber_g=0.0,
    )

    result = lookup_nutrition("grilled_salmon", db_session)

    assert result is not None
    assert result["food_name"] == "Grilled Salmon"
    assert result["calories"] == 412.0
    assert result["protein_g"] == 40.0
    assert result["fat_g"] == 27.0


def test_fuzzy_match_returns_partial(db_session: Session) -> None:
    # Aucun match exact pour "grilled_salmon" : seul un nom contenant
    # tous les mots-cles ("grilled" et "salmon") est present.
    _insert(
        db_session,
        food_name="Salmon Filet, Grilled with Herbs",
        calories=350.0,
        protein_g=35.0,
        carbs_g=2.0,
        fat_g=22.0,
        fiber_g=0.5,
    )
    _insert(db_session, food_name="Tuna Steak", calories=200.0)

    result = lookup_nutrition("grilled_salmon", db_session)

    assert result is not None
    assert result["food_name"] == "Salmon Filet, Grilled with Herbs"
    assert result["calories"] == 350.0


def test_total_miss_returns_none(db_session: Session) -> None:
    # Aucun aliment ne contient ni "grilled" ni "salmon".
    _insert(db_session, food_name="Apple Pie")
    _insert(db_session, food_name="Caesar Salad")

    result = lookup_nutrition("grilled_salmon", db_session)

    assert result is None


def test_empty_table_returns_none(db_session: Session) -> None:
    result = lookup_nutrition("grilled_salmon", db_session)

    assert result is None
