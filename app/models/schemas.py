from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class HealthGoal(str, Enum):
    weight_loss = "weight_loss"
    muscle_gain = "muscle_gain"
    balance = "balance"
    sport_performance = "sport_performance"


# MealAnalysis

class MealAnalysisItem(BaseModel):
    id: int
    detected_foods: list[dict[str, Any]]
    macros: dict[str, float]
    recommendations: list[str]
    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedAnalysesResponse(BaseModel):
    items: list[MealAnalysisItem]
    total: int
    limit: int
    offset: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
                    {
                        "id": 42,
                        "detected_foods": [
                            {"label": "pizza", "confidence": 0.85, "nutrition": {}}
                        ],
                        "macros": {"calories": 1300, "protein_g": 30, "carbs_g": 160, "fat_g": 50},
                        "recommendations": ["Reduis la portion au prochain repas."],
                        "created_at": "2026-04-29T10:15:00",
                    }
                ],
                "total": 137,
                "limit": 20,
                "offset": 0,
            }
        }
    }


# NutritionGoal

class NutritionGoalRequest(BaseModel):
    health_goal: HealthGoal | None = None
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


# Generate-meal-plan : contrat d'API (issue NUT-9).


class DietType(str, Enum):
    omnivore = "omnivore"
    vegetarien = "vegetarien"
    vegan = "vegan"
    sans_gluten = "sans_gluten"


class MealPlanRequest(BaseModel):
    health_goal: HealthGoal | None = None
    allergies: list[str] = []
    budget_eur_per_day: float | None = Field(default=None, ge=0)
    diet_type: DietType
    duration_days: int = Field(default=7, ge=1, le=30)


class MealPlanResponse(BaseModel):
    plan_id: int
    fallback: bool
    days: list[MealDay]


class MealPlanHistoryItem(BaseModel):
    id: int
    objective: str | None
    constraints: dict[str, Any]
    plan: dict[str, Any]
    generated_at: datetime

    model_config = {"from_attributes": True}


class PaginatedPlansResponse(BaseModel):
    items: list[MealPlanHistoryItem]
    total: int
    limit: int
    offset: int

    model_config = {
        "json_schema_extra": {
            "example": {
                "items": [
                    {
                        "id": 42,
                        "objective": "weight_loss",
                        "constraints": {
                            "diet_type": "vegan",
                            "duration_days": 7,
                            "allergies": [],
                        },
                        "plan": {
                            "days": [{"day": 1, "meals": []}],
                        },
                        "generated_at": "2026-04-29T10:15:00",
                    }
                ],
                "total": 17,
                "limit": 20,
                "offset": 0,
            }
        }
    }


# LLM client : inputs / outputs (issue NUT-7).


class Imbalance(str, Enum):
    balanced = "balanced"
    protein_low = "protein_low"
    protein_high = "protein_high"
    carbs_low = "carbs_low"
    carbs_high = "carbs_high"
    fat_low = "fat_low"
    fat_high = "fat_high"
    calories_low = "calories_low"
    calories_high = "calories_high"


# Inputs pour la generation de plan repas. allergies est triee dans
# canonicalize_inputs avant calcul du hash.
class PlanInputs(BaseModel):
    user_id: int
    objective: str
    duration_days: int = 7
    diet_type: str | None = None
    allergies: list[str] = Field(default_factory=list)
    budget_per_day: Decimal | None = None
    calories_target: int | None = None


class RecommendationContext(BaseModel):
    user_id: int
    imbalance: Imbalance
    health_goal: HealthGoal
