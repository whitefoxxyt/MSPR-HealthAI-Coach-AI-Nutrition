from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel


# MealAnalysis

class MealAnalysisResponse(BaseModel):
    id: int
    user_id: int
    photo_url: str | None
    detected_foods: list
    macros: dict
    confidence_scores: dict
    created_at: datetime

    model_config = {"from_attributes": True}


# MealPlan

class MealPlanRequest(BaseModel):
    user_id: int
    objective: str | None = None
    constraints: dict = {}


class MealPlanResponse(BaseModel):
    id: int
    user_id: int
    plan: dict
    objective: str | None
    constraints: dict
    generated_at: datetime

    model_config = {"from_attributes": True}


# NutritionGoal

class NutritionGoalRequest(BaseModel):
    calories_target: int | None = None
    protein_g: Decimal | None = None
    carbs_g: Decimal | None = None
    fat_g: Decimal | None = None
    allergies: list[str] = []
    diet_type: str | None = None


class NutritionGoalResponse(NutritionGoalRequest):
    user_id: int

    model_config = {"from_attributes": True}


# Plans repas fallback (issue NUT-8). Cf. POST /api/v1/generate-meal-plan.

class MealMacros(BaseModel):
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float


class Meal(BaseModel):
    name: str
    macros: MealMacros
    ingredients: list[str]
    est_budget_eur: float
    prep_time_min: int


class MealDay(BaseModel):
    day: int
    meals: list[Meal]


class FallbackMealPlan(BaseModel):
    fallback: bool
    days: list[MealDay]
