from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel, Field


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


# LLM client : inputs / outputs


class HealthGoal(str, Enum):
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    BALANCE = "balance"
    SPORT_PERFORMANCE = "sport_performance"


class Imbalance(str, Enum):
    BALANCED = "balanced"
    PROTEIN_LOW = "protein_low"
    PROTEIN_HIGH = "protein_high"
    CARBS_LOW = "carbs_low"
    CARBS_HIGH = "carbs_high"
    FAT_LOW = "fat_low"
    FAT_HIGH = "fat_high"
    CALORIES_LOW = "calories_low"
    CALORIES_HIGH = "calories_high"


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


class MealEntry(BaseModel):
    name: str
    ingredients: list[str]
    calories: int
    protein_g: float = 0.0
    carbs_g: float = 0.0
    fat_g: float = 0.0


class DayPlan(BaseModel):
    day: int
    meals: list[MealEntry]


class MealPlan(BaseModel):
    days: list[DayPlan]
    total_calories: int
    fallback: bool = False


class RecommendationContext(BaseModel):
    user_id: int
    imbalance: Imbalance
    health_goal: HealthGoal
