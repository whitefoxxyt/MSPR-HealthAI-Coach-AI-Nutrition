from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthGoal(str, Enum):
    weight_loss = "weight_loss"
    muscle_gain = "muscle_gain"
    balance = "balance"
    sport_performance = "sport_performance"


class Gender(str, Enum):
    male = "male"
    female = "female"


class ActivityLevel(str, Enum):
    sedentary = "sedentary"
    light = "light"
    moderate = "moderate"
    active = "active"
    very_active = "very_active"


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
                        "macros": {
                            "calories": 1300,
                            "protein_g": 30,
                            "carbs_g": 160,
                            "fat_g": 50,
                        },
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
    # Biometrie (V12) : entree pour le calcul du TDEE par nutrition_engine.
    # gt=0 : Mifflin-St Jeor n'a pas de sens sur des valeurs nulles ou negatives,
    # on rejette des l'API plutot que de calculer un TDEE absurde plus tard.
    gender: Gender | None = None
    age: int | None = Field(default=None, gt=0)
    weight_kg: Decimal | None = Field(default=None, gt=0)
    height_cm: Decimal | None = Field(default=None, gt=0)
    activity_level: ActivityLevel | None = None


class NutritionGoalResponse(NutritionGoalRequest):
    user_id: int

    model_config = {"from_attributes": True}


# GET /me/macros : cible journaliere derivee du profil + objectif sante (slice 2 PRD #45).


class MacroTargetsView(BaseModel):
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


class MeMacrosResponse(BaseModel):
    profile_completion_required: bool
    missing_fields: list[str] = []
    tdee: float | None = None
    macros: MacroTargetsView | None = None


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
    # DeCRIM-light : sortie de la boucle retry / validation / fallback (slice 7).
    # full           : plan LLM 100% conforme aux contraintes.
    # partial_budget : allergies + regime OK, budget legerement depasse.
    # static_fallback: bascule sur le plan statique (LLM infaisable ou injoignable).
    compliance_status: Literal["full", "partial_budget", "static_fallback"] = "full"
    compliance_warnings: list[str] = Field(default_factory=list)


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


# Imbalance tags structures (issue #51, slice 5 PRD #45).
# Remplacent l'ancien enum Imbalance + phrases. Le detector emet des tags ;
# imbalance_to_text les convertit en francais ; le LLM en fait une synthese.


class Nutrient(str, Enum):
    calories = "calories"
    protein_g = "protein_g"
    carbs_g = "carbs_g"
    fat_g = "fat_g"
    fibers_g = "fibers_g"
    saturated_fat_g = "saturated_fat_g"


class ImbalanceStatus(str, Enum):
    excess = "excess"
    deficit = "deficit"
    ok = "ok"


class ImbalanceTag(BaseModel):
    nutrient: Nutrient
    status: ImbalanceStatus
    # Ecart relatif a la cible : (actual - target) / target. Positif si excess,
    # negatif si deficit. ok -> 0 (le tag n'est generalement pas emis dans ce cas).
    delta_pct: float
    target_value: float
    actual_value: float
    unit: str


# Reponse de POST /analyze-meal (issue #51, slice 5 PRD #45).


class MealAnalysisResponse(BaseModel):
    analysis_id: int
    detected_foods: list[dict[str, Any]]
    macros: dict[str, float]
    imbalances: list[ImbalanceTag]
    imbalances_text: list[str]
    recommendations: list[str]
    fallback: bool
    profile_completion_required: bool
    missing_fields: list[str] = []


# Tailles de portion PNNS (issue NUT-49). Cf. app/data/portion_sizes.py.


class ServingSizeLabel(str, Enum):
    small = "small"
    medium = "medium"
    large = "large"


class ServingSize(BaseModel):
    # frozen=True : portion_sizes.get_serving_sizes renvoie list(...) (defense
    # niveau liste) mais les ServingSize sont partagees avec le cache global
    # _PNNS_SERVING_SIZES. Sans frozen, un caller peut faire `p[0].grams = 999`
    # et corrompre la table de reference.
    model_config = {"frozen": True}

    label: ServingSizeLabel
    grams: int = Field(gt=0)
    description: str | None = None
