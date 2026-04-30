from __future__ import annotations

import pytest

from app.models.schemas import HealthGoal
from app.services.nutrition_engine import (
    ActivityLevel,
    Gender,
    IncompleteProfile,
    MealType,
    UserProfile,
    build_user_profile,
    compute_bmr,
    compute_macro_targets,
    compute_meal_targets,
    compute_tdee,
)


def test_compute_bmr_male_mifflin_st_jeor_textbook_case() -> None:
    # Mifflin-St Jeor homme : 10*80 + 6.25*180 - 5*30 + 5 = 1780 kcal/j.
    profile = UserProfile(
        gender=Gender.male,
        age=30,
        weight_kg=80.0,
        height_cm=180.0,
        activity_level=ActivityLevel.moderate,
    )

    assert compute_bmr(profile) == pytest.approx(1780.0)


def test_compute_bmr_female_mifflin_st_jeor_textbook_case() -> None:
    # Mifflin-St Jeor femme : 10*60 + 6.25*165 - 5*25 - 161 = 1345.25 kcal/j.
    profile = UserProfile(
        gender=Gender.female,
        age=25,
        weight_kg=60.0,
        height_cm=165.0,
        activity_level=ActivityLevel.light,
    )

    assert compute_bmr(profile) == pytest.approx(1345.25)


@pytest.mark.parametrize(
    ("activity", "expected_tdee"),
    [
        # BMR homme reference = 1780 kcal/j (cf. test compute_bmr male).
        (ActivityLevel.sedentary, 1780.0 * 1.2),
        (ActivityLevel.light, 1780.0 * 1.375),
        (ActivityLevel.moderate, 1780.0 * 1.55),
        (ActivityLevel.active, 1780.0 * 1.725),
        (ActivityLevel.very_active, 1780.0 * 1.9),
    ],
)
def test_compute_tdee_applies_activity_factor(
    activity: ActivityLevel, expected_tdee: float
) -> None:
    profile = UserProfile(
        gender=Gender.male,
        age=30,
        weight_kg=80.0,
        height_cm=180.0,
        activity_level=activity,
    )

    assert compute_tdee(profile) == pytest.approx(expected_tdee)


def test_compute_macro_targets_balance_uses_50_20_30_split() -> None:
    # health_goal=balance : 50 % glucides, 20 % proteines, 30 % lipides.
    # TDEE 2000 kcal :
    #   prot  = 0.20 * 2000 / 4 = 100 g
    #   carbs = 0.50 * 2000 / 4 = 250 g
    #   fat   = 0.30 * 2000 / 9 ~= 66.67 g
    macros = compute_macro_targets(tdee=2000.0, health_goal=HealthGoal.balance)

    assert macros.calories == pytest.approx(2000.0)
    assert macros.protein_g == pytest.approx(100.0)
    assert macros.carbs_g == pytest.approx(250.0)
    assert macros.fat_g == pytest.approx(2000.0 * 0.30 / 9.0)


@pytest.mark.parametrize(
    ("health_goal", "carbs_pct", "protein_pct", "fat_pct"),
    [
        (HealthGoal.balance, 0.50, 0.20, 0.30),
        (HealthGoal.weight_loss, 0.40, 0.30, 0.30),
        (HealthGoal.muscle_gain, 0.45, 0.30, 0.25),
        (HealthGoal.sport_performance, 0.55, 0.20, 0.25),
    ],
)
def test_compute_macro_targets_uses_distinct_splits_per_goal(
    health_goal: HealthGoal,
    carbs_pct: float,
    protein_pct: float,
    fat_pct: float,
) -> None:
    tdee = 2400.0
    macros = compute_macro_targets(tdee=tdee, health_goal=health_goal)

    assert macros.calories == pytest.approx(tdee)
    assert macros.protein_g == pytest.approx(tdee * protein_pct / 4.0)
    assert macros.carbs_g == pytest.approx(tdee * carbs_pct / 4.0)
    assert macros.fat_g == pytest.approx(tdee * fat_pct / 9.0)


def test_compute_meal_targets_breakfast_applies_25_pct_quota() -> None:
    profile = UserProfile(
        gender=Gender.male,
        age=30,
        weight_kg=80.0,
        height_cm=180.0,
        activity_level=ActivityLevel.moderate,
    )
    # TDEE = 1780 * 1.55 = 2759 kcal/j ; breakfast = 25 % -> 689.75 kcal cible.
    # Repartition balance sur la part repas : 50 % glucides, 20 % prot, 30 % lip.
    daily_kcal = 1780.0 * 1.55
    quota = 0.25

    targets = compute_meal_targets(
        profile=profile,
        meal_type=MealType.breakfast,
        health_goal=HealthGoal.balance,
    )

    assert targets.calories == pytest.approx(daily_kcal * quota)
    assert targets.protein_g == pytest.approx(daily_kcal * quota * 0.20 / 4.0)
    assert targets.carbs_g == pytest.approx(daily_kcal * quota * 0.50 / 4.0)
    assert targets.fat_g == pytest.approx(daily_kcal * quota * 0.30 / 9.0)


@pytest.mark.parametrize(
    ("meal_type", "expected_quota"),
    [
        (MealType.breakfast, 0.25),
        (MealType.lunch, 0.35),
        (MealType.dinner, 0.30),
        (MealType.snack, 0.10),
    ],
)
def test_compute_meal_targets_applies_quota_per_meal_type(
    meal_type: MealType, expected_quota: float
) -> None:
    profile = UserProfile(
        gender=Gender.male,
        age=30,
        weight_kg=80.0,
        height_cm=180.0,
        activity_level=ActivityLevel.moderate,
    )
    daily_kcal = 1780.0 * 1.55  # TDEE moderate

    targets = compute_meal_targets(
        profile=profile,
        meal_type=meal_type,
        health_goal=HealthGoal.balance,
    )

    assert targets.calories == pytest.approx(daily_kcal * expected_quota)


def test_meal_quotas_sum_to_100_percent() -> None:
    # Garde-fou : 25 + 35 + 30 + 10 = 100. Si quelqu'un retouche un quota
    # en oubliant les autres, ce test casse.
    from app.services.nutrition_engine import _MEAL_QUOTAS

    assert sum(_MEAL_QUOTAS.values()) == pytest.approx(1.0)


def test_compute_meal_targets_falls_back_to_tdee_div_4_when_meal_type_absent() -> None:
    # Issue 47 : "fallback TDEE/4 si meal_type absent".
    profile = UserProfile(
        gender=Gender.male,
        age=30,
        weight_kg=80.0,
        height_cm=180.0,
        activity_level=ActivityLevel.moderate,
    )
    daily_kcal = 1780.0 * 1.55

    targets = compute_meal_targets(
        profile=profile,
        meal_type=None,
        health_goal=HealthGoal.balance,
    )

    assert targets.calories == pytest.approx(daily_kcal / 4.0)


def test_anses_rnp_constants_match_official_values() -> None:
    # Reperes ANSES (rapport "Actualisation des reperes du PNNS", 2016 ;
    # avis 2017-SA-0142). Sources figees pour `imbalance_detector` (slice 5).
    from app import config

    assert config.RNP_PROTEIN_G_PER_KG == pytest.approx(0.83)
    assert config.RNP_FIBER_G_PER_DAY == 30
    assert config.RNP_AGS_PERCENT_OF_AET_MAX == pytest.approx(0.12)
    assert config.RNP_TOTAL_SUGARS_G_MAX == 100


def test_build_user_profile_with_no_source_returns_all_fields_missing() -> None:
    result = build_user_profile(None)

    assert isinstance(result, IncompleteProfile)
    assert set(result.missing_fields) == {
        "gender",
        "age",
        "weight_kg",
        "height_cm",
        "activity_level",
    }


def test_build_user_profile_with_partial_source_lists_only_missing_fields() -> None:
    from types import SimpleNamespace

    source = SimpleNamespace(
        gender="male",
        age=30,
        weight_kg=None,
        height_cm=180.0,
        activity_level=None,
    )

    result = build_user_profile(source)

    assert isinstance(result, IncompleteProfile)
    assert set(result.missing_fields) == {"weight_kg", "activity_level"}


def test_build_user_profile_with_complete_source_returns_user_profile() -> None:
    from types import SimpleNamespace

    source = SimpleNamespace(
        gender="female",
        age=25,
        weight_kg=60.0,
        height_cm=165.0,
        activity_level="light",
    )

    result = build_user_profile(source)

    assert isinstance(result, UserProfile)
    assert result.gender is Gender.female
    assert result.age == 25
    assert result.weight_kg == pytest.approx(60.0)
    assert result.height_cm == pytest.approx(165.0)
    assert result.activity_level is ActivityLevel.light
