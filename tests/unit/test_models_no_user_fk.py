from __future__ import annotations

# La table users a ete droppee par MSPR-DB V7. Les FK declarees au niveau
# SQLAlchemy pointaient dans le vide et auraient casse tout create_all().
# Ces tests verrouillent la suppression et empechent la regression.

from app.db.models import MealAnalysis, MealPlan, NutritionGoal


def test_meal_analysis_user_id_has_no_fk() -> None:
    assert MealAnalysis.__table__.c.user_id.foreign_keys == set()


def test_meal_plan_user_id_has_no_fk() -> None:
    assert MealPlan.__table__.c.user_id.foreign_keys == set()


def test_nutrition_goal_user_id_has_no_fk() -> None:
    assert NutritionGoal.__table__.c.user_id.foreign_keys == set()


def test_user_id_columns_remain_not_nullable() -> None:
    assert MealAnalysis.__table__.c.user_id.nullable is False
    assert MealPlan.__table__.c.user_id.nullable is False
    # NutritionGoal.user_id est primary_key donc implicitement NOT NULL.
    assert NutritionGoal.__table__.c.user_id.primary_key is True
    assert NutritionGoal.__table__.c.user_id.nullable is False
