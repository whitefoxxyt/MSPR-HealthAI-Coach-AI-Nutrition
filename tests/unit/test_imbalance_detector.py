from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.db.models import NutritionGoal
from app.models.schemas import HealthGoal, ImbalanceStatus, ImbalanceTag, Nutrient
from app.services.imbalance_detector import detect, imbalance_to_text
from app.services.nutrition_engine import MealType


def _full_goal(**overrides) -> NutritionGoal:
    """NutritionGoal avec biometrie complete (TDEE calculable)."""
    base = {
        "user_id": 1,
        "gender": "male",
        "age": 30,
        "weight_kg": 80.0,
        "height_cm": 180.0,
        "activity_level": "moderate",
        "health_goal": "balance",
    }
    base.update(overrides)
    return NutritionGoal(**base)


# Cycle 2 : profil incomplet -> []


def test_detect_returns_empty_when_profile_is_none() -> None:
    tags = detect(
        meal_macros={"calories": 700, "protein_g": 30, "carbs_g": 80, "fat_g": 25},
        profile=None,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert tags == []


def test_detect_returns_empty_when_profile_missing_biometric_field() -> None:
    # Sans weight_kg : Mifflin-St Jeor n'est pas calculable -> liste vide.
    incomplete = SimpleNamespace(
        gender="male",
        age=30,
        weight_kg=None,
        height_cm=180.0,
        activity_level="moderate",
    )

    tags = detect(
        meal_macros={"calories": 5000},
        profile=incomplete,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert tags == []


def test_detect_returns_empty_when_meal_macros_empty() -> None:
    # Sans macros mesurees, rien a comparer.
    profile = _full_goal()

    tags = detect(
        meal_macros={},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert tags == []


# Cycle 3 : calories +/- 20 % symetrique


def test_detect_emits_calories_excess_when_meal_above_120_pct_target() -> None:
    profile = _full_goal()
    # TDEE moderate homme 30 ans 80 kg 180 cm = 1780 * 1.55 = 2759 kcal/j
    # Lunch quota 35 % -> cible 965.65 kcal. +50 % -> 1448 kcal -> excess.
    actual = 1450.0

    tags = detect(
        meal_macros={"calories": actual},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    calories_tags = [t for t in tags if t.nutrient is Nutrient.calories]
    assert len(calories_tags) == 1
    tag = calories_tags[0]
    assert tag.status is ImbalanceStatus.excess
    assert tag.actual_value == pytest.approx(actual)
    assert tag.target_value == pytest.approx(2759.0 * 0.35, rel=0.01)
    assert tag.delta_pct > 0.20
    assert tag.unit == "kcal"


def test_detect_emits_calories_deficit_when_meal_below_80_pct_target() -> None:
    profile = _full_goal()
    # Lunch quota 35 % -> cible ~965 kcal. 50 % en dessous (482 kcal) -> deficit.
    actual = 482.0

    tags = detect(
        meal_macros={"calories": actual},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    calories_tags = [t for t in tags if t.nutrient is Nutrient.calories]
    assert len(calories_tags) == 1
    tag = calories_tags[0]
    assert tag.status is ImbalanceStatus.deficit
    assert tag.delta_pct < -0.20


def test_detect_does_not_emit_calories_when_meal_within_tolerance_band() -> None:
    profile = _full_goal()
    # Cible ~965 kcal pour lunch ; +/-20 % -> bande [772, 1158]. 1000 dedans.
    tags = detect(
        meal_macros={"calories": 1000.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert all(t.nutrient is not Nutrient.calories for t in tags)


def test_detect_calories_delta_pct_matches_relative_difference() -> None:
    profile = _full_goal()
    # Cible lunch ~ 2759 * 0.35 = 965.65 kcal. Actual = 1500 -> delta = 0.554.
    actual = 1500.0

    tags = detect(
        meal_macros={"calories": actual},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    tag = next(t for t in tags if t.nutrient is Nutrient.calories)
    expected_target = 2759.0 * 0.35
    expected_delta = (actual - expected_target) / expected_target
    assert tag.delta_pct == pytest.approx(expected_delta, rel=0.01)


# Cycle 4 : macros (protein / carbs / fat) symetrique +/- 20 %


def test_detect_emits_protein_deficit_when_below_80_pct_target() -> None:
    profile = _full_goal()
    # balance : protein_pct = 20 %. Lunch quota 35 %. Pour TDEE 2759 :
    # cible protein lunch = 2759 * 0.20 / 4 * 0.35 = 48.28 g.
    # Actual 30 g -> delta = -0.379 (deficit).
    tags = detect(
        meal_macros={"protein_g": 30.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    protein_tags = [t for t in tags if t.nutrient is Nutrient.protein_g]
    assert len(protein_tags) == 1
    assert protein_tags[0].status is ImbalanceStatus.deficit
    assert protein_tags[0].unit == "g"


def test_detect_emits_carbs_excess_when_above_120_pct_target() -> None:
    profile = _full_goal()
    # balance : carbs_pct = 50 %. Lunch quota 35 %. Cible carbs ~ 120.7 g.
    # Actual 200 g -> delta ~ +0.66 (excess).
    tags = detect(
        meal_macros={"carbs_g": 200.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    carbs_tags = [t for t in tags if t.nutrient is Nutrient.carbs_g]
    assert len(carbs_tags) == 1
    assert carbs_tags[0].status is ImbalanceStatus.excess


def test_detect_emits_fat_deficit_when_below_80_pct_target() -> None:
    profile = _full_goal()
    # balance : fat_pct = 30 %. Lunch quota 35 %. TDEE 2759 :
    # cible fat lunch = 2759 * 0.30 / 9 * 0.35 = 32.2 g.
    # Actual 10 g -> delta -0.69 (deficit).
    tags = detect(
        meal_macros={"fat_g": 10.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    fat_tags = [t for t in tags if t.nutrient is Nutrient.fat_g]
    assert len(fat_tags) == 1
    assert fat_tags[0].status is ImbalanceStatus.deficit


def test_detect_aligned_meal_returns_no_macro_tags() -> None:
    profile = _full_goal()
    # Repas approximativement aligne sur la cible lunch balance.
    # Cible : 965 kcal / 48 g prot / 121 g carbs / 32 g fat (a +/- 20 %).
    tags = detect(
        meal_macros={
            "calories": 950.0,
            "protein_g": 48.0,
            "carbs_g": 120.0,
            "fat_g": 32.0,
        },
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert tags == []


# Cycle 5 : AGS (acides gras satures) plafond seul


def test_detect_emits_saturated_fat_excess_above_12_pct_aet() -> None:
    profile = _full_goal()
    # Plafond ANSES : AGS <= 12 % AET. Lunch quota 35 % du TDEE 2759 = 965.65 kcal.
    # Cible AGS = 965.65 * 0.12 / 9 = 12.88 g. Actual 25 g -> excess.
    tags = detect(
        meal_macros={"saturated_fat_g": 25.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    ags_tags = [t for t in tags if t.nutrient is Nutrient.saturated_fat_g]
    assert len(ags_tags) == 1
    tag = ags_tags[0]
    assert tag.status is ImbalanceStatus.excess
    assert tag.unit == "g"
    assert tag.target_value == pytest.approx(965.65 * 0.12 / 9.0, rel=0.01)
    assert tag.actual_value == pytest.approx(25.0)


def test_detect_does_not_emit_saturated_fat_when_below_target() -> None:
    profile = _full_goal()
    # 5 g d'AGS contre cible ~12.9 g -> pas de deficit (asymetrique, plafond seul).
    tags = detect(
        meal_macros={"saturated_fat_g": 5.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert all(t.nutrient is not Nutrient.saturated_fat_g for t in tags)


def test_detect_does_not_emit_saturated_fat_when_at_zero() -> None:
    # Cas extreme : AGS = 0. Ne doit pas declencher de deficit.
    profile = _full_goal()
    tags = detect(
        meal_macros={"saturated_fat_g": 0.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert all(t.nutrient is not Nutrient.saturated_fat_g for t in tags)


# Cycle 6 : fibres deficit seul


def test_detect_emits_fibers_deficit_when_below_80_pct_target() -> None:
    profile = _full_goal()
    # RNP 30 g/j * lunch quota 0.35 = 10.5 g cible. Actual 4 g -> delta -0.62 -> deficit.
    tags = detect(
        meal_macros={"fibers_g": 4.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    fiber_tags = [t for t in tags if t.nutrient is Nutrient.fibers_g]
    assert len(fiber_tags) == 1
    tag = fiber_tags[0]
    assert tag.status is ImbalanceStatus.deficit
    assert tag.unit == "g"
    assert tag.target_value == pytest.approx(30.0 * 0.35)
    assert tag.actual_value == pytest.approx(4.0)


def test_detect_does_not_emit_fibers_when_above_target() -> None:
    profile = _full_goal()
    # 15 g pour cible 10.5 g -> excess, mais asymetrique : pas de tag emis.
    tags = detect(
        meal_macros={"fibers_g": 15.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert all(t.nutrient is not Nutrient.fibers_g for t in tags)


def test_detect_accepts_legacy_fiber_g_key_for_fibers() -> None:
    # nutrition_lookup expose la macro sous la cle 'fiber_g' (singulier).
    # On accepte cette cle en plus de 'fibers_g' pour ne pas casser le pipeline.
    profile = _full_goal()
    tags = detect(
        meal_macros={"fiber_g": 2.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert any(t.nutrient is Nutrient.fibers_g for t in tags)


def test_detect_skips_macros_not_present_in_meal() -> None:
    profile = _full_goal()
    # Seulement calories : pas de tag pour les macros absentes.
    tags = detect(
        meal_macros={"calories": 1500.0},
        profile=profile,
        meal_type=MealType.lunch,
        health_goal=HealthGoal.balance,
    )

    assert {t.nutrient for t in tags} == {Nutrient.calories}


# Cycle 7 : imbalance_to_text


def _tag(
    nutrient: Nutrient,
    status: ImbalanceStatus,
    delta_pct: float,
    target: float,
    actual: float,
    unit: str,
) -> ImbalanceTag:
    return ImbalanceTag(
        nutrient=nutrient,
        status=status,
        delta_pct=delta_pct,
        target_value=target,
        actual_value=actual,
        unit=unit,
    )


def test_imbalance_to_text_calories_excess_mentions_meal_calories_and_target() -> None:
    tag = _tag(Nutrient.calories, ImbalanceStatus.excess, 0.50, 700.0, 1050.0, "kcal")

    text = imbalance_to_text(tag)

    assert "calories" in text.lower() or "calorique" in text.lower()
    assert "1050" in text  # actual
    assert "700" in text  # target
    assert "kcal" in text


def test_imbalance_to_text_protein_deficit_mentions_proteines() -> None:
    tag = _tag(Nutrient.protein_g, ImbalanceStatus.deficit, -0.40, 50.0, 30.0, "g")

    text = imbalance_to_text(tag)

    assert "protein" in text.lower() or "protéin" in text.lower()
    # Marqueur deficit (mots usuels en francais).
    assert any(
        w in text.lower() for w in ("faible", "manque", "insuffisant", "deficit")
    )


def test_imbalance_to_text_saturated_fat_excess_mentions_ags_or_satures() -> None:
    tag = _tag(Nutrient.saturated_fat_g, ImbalanceStatus.excess, 0.95, 12.9, 25.0, "g")

    text = imbalance_to_text(tag)

    lowered = text.lower()
    assert "satur" in lowered or "ags" in lowered


def test_imbalance_to_text_fibers_deficit_mentions_fibres() -> None:
    tag = _tag(Nutrient.fibers_g, ImbalanceStatus.deficit, -0.62, 10.5, 4.0, "g")

    text = imbalance_to_text(tag)

    assert "fibre" in text.lower()


def test_imbalance_to_text_is_deterministic_for_same_tag() -> None:
    tag = _tag(Nutrient.calories, ImbalanceStatus.excess, 0.30, 700.0, 910.0, "kcal")

    a = imbalance_to_text(tag)
    b = imbalance_to_text(tag)

    assert a == b
