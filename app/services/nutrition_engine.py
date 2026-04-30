from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.models.schemas import ActivityLevel, Gender, HealthGoal


@dataclass(frozen=True)
class UserProfile:
    gender: Gender
    age: int
    weight_kg: float
    height_cm: float
    activity_level: ActivityLevel


_ACTIVITY_FACTORS: dict[ActivityLevel, float] = {
    ActivityLevel.sedentary: 1.2,
    ActivityLevel.light: 1.375,
    ActivityLevel.moderate: 1.55,
    ActivityLevel.active: 1.725,
    ActivityLevel.very_active: 1.9,
}


# Repartition (% glucides, % proteines, % lipides) par objectif sante.
# - balance         : reperes ANSES generaux (50/20/30)
# - weight_loss     : proteines elevees pour preserver la masse maigre
# - muscle_gain     : surplus glucidique modere + proteines elevees
# - sport_performance : glucides eleves pour la resynthese du glycogene
_MACRO_SPLITS: dict[HealthGoal, tuple[float, float, float]] = {
    HealthGoal.balance: (0.50, 0.20, 0.30),
    HealthGoal.weight_loss: (0.40, 0.30, 0.30),
    HealthGoal.muscle_gain: (0.45, 0.30, 0.25),
    HealthGoal.sport_performance: (0.55, 0.20, 0.25),
}


class MealType(str, Enum):
    breakfast = "breakfast"
    lunch = "lunch"
    dinner = "dinner"
    snack = "snack"


# Reperes PNNS / pratique courante : repartition energetique sur la journee.
_MEAL_QUOTAS: dict[MealType, float] = {
    MealType.breakfast: 0.25,
    MealType.lunch: 0.35,
    MealType.dinner: 0.30,
    MealType.snack: 0.10,
}


@dataclass(frozen=True)
class MacroTargets:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


@dataclass(frozen=True)
class MealTargets:
    calories: float
    protein_g: float
    carbs_g: float
    fat_g: float


@dataclass(frozen=True)
class IncompleteProfile:
    missing_fields: list[str]


_PROFILE_FIELDS: tuple[str, ...] = (
    "gender",
    "age",
    "weight_kg",
    "height_cm",
    "activity_level",
)


def compute_bmr(profile: UserProfile) -> float:
    # Mifflin-St Jeor :
    #   homme  : 10*kg + 6.25*cm - 5*age + 5
    #   femme  : 10*kg + 6.25*cm - 5*age - 161
    base = 10.0 * profile.weight_kg + 6.25 * profile.height_cm - 5.0 * profile.age
    if profile.gender is Gender.male:
        return base + 5.0
    return base - 161.0


def compute_tdee(profile: UserProfile) -> float:
    return compute_bmr(profile) * _ACTIVITY_FACTORS[profile.activity_level]


def compute_macro_targets(tdee: float, health_goal: HealthGoal) -> MacroTargets:
    carbs_pct, protein_pct, fat_pct = _MACRO_SPLITS[health_goal]
    # 4 kcal/g pour proteines et glucides, 9 kcal/g pour lipides.
    return MacroTargets(
        calories=tdee,
        protein_g=tdee * protein_pct / 4.0,
        carbs_g=tdee * carbs_pct / 4.0,
        fat_g=tdee * fat_pct / 9.0,
    )


def build_user_profile(source: object | None) -> UserProfile | IncompleteProfile:
    # Source = NutritionGoal ORM ou tout objet exposant les 5 attributs profil.
    # Champ manquant = absent ou explicitement None ; on agglome dans
    # missing_fields pour permettre au front de pointer ce qu'il manque.
    if source is None:
        return IncompleteProfile(missing_fields=list(_PROFILE_FIELDS))
    missing = [name for name in _PROFILE_FIELDS if getattr(source, name, None) is None]
    if missing:
        return IncompleteProfile(missing_fields=missing)
    return UserProfile(
        gender=Gender(source.gender),
        age=int(source.age),
        weight_kg=float(source.weight_kg),
        height_cm=float(source.height_cm),
        activity_level=ActivityLevel(source.activity_level),
    )


def compute_meal_targets(
    profile: UserProfile,
    meal_type: MealType | None,
    health_goal: HealthGoal,
) -> MealTargets:
    # Fallback : meal_type absent -> repartition uniforme sur 4 prises.
    quota = _MEAL_QUOTAS[meal_type] if meal_type is not None else 0.25
    daily = compute_macro_targets(compute_tdee(profile), health_goal)
    return MealTargets(
        calories=daily.calories * quota,
        protein_g=daily.protein_g * quota,
        carbs_g=daily.carbs_g * quota,
        fat_g=daily.fat_g * quota,
    )
