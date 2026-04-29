from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum

from pydantic import BaseModel


class HealthGoal(str, Enum):
    weight_loss = "weight_loss"
    muscle_gain = "muscle_gain"
    balance = "balance"
    sport_performance = "sport_performance"


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
