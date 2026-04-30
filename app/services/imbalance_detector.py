from __future__ import annotations

from app import config
from app.db.models import NutritionGoal
from app.models.schemas import (
    HealthGoal,
    ImbalanceStatus,
    ImbalanceTag,
    Nutrient,
)
from app.services.nutrition_engine import (
    IncompleteProfile,
    MealType,
    build_user_profile,
    compute_meal_targets,
    meal_quota,
)

# Detection des desequilibres macros par rapport a une cible repas issue de
# nutrition_engine (Mifflin-St Jeor + RNP ANSES). Les phrases francaises sont
# generees par imbalance_to_text (templates Python) ; le LLM fait la synthese.

# Tolerance symetrique appliquee a calories + macros (P/C/F).
_MACRO_TOLERANCE = 0.20

# Plafond AGS : 12 % de l'AET (apport energetique total) selon ANSES.
# AGS apporte 9 kcal/g.
_KCAL_PER_FAT_GRAM = 9.0


def detect(
    meal_macros: dict[str, float],
    profile: NutritionGoal | None,
    meal_type: MealType | None,
    health_goal: HealthGoal,
) -> list[ImbalanceTag]:
    """Compare les macros du repas aux cibles personnalisees et retourne des tags.

    Profil incomplet ou macros vides -> []. La decision de notifier le client
    (profile_completion_required) est deferee a meal_analysis_orchestrator.

    Tolerances :
    - calories, protein_g, carbs_g, fat_g : symetrique +/- 20 % autour de la cible
    - saturated_fat_g : plafond seul (excess > 12 % AET)
    - fibers_g : deficit seul (< 80 % de la part repas du RNP 30 g/j)
    """
    if not meal_macros:
        return []
    user_profile = build_user_profile(profile)
    if isinstance(user_profile, IncompleteProfile):
        return []

    targets = compute_meal_targets(user_profile, meal_type, health_goal)
    tags: list[ImbalanceTag] = []

    # Symetrique +/- 20 % sur calories et macros.
    for nutrient, actual_key, target_value, unit in (
        (Nutrient.calories, "calories", targets.calories, "kcal"),
        (Nutrient.protein_g, "protein_g", targets.protein_g, "g"),
        (Nutrient.carbs_g, "carbs_g", targets.carbs_g, "g"),
        (Nutrient.fat_g, "fat_g", targets.fat_g, "g"),
    ):
        actual = meal_macros.get(actual_key)
        if actual is None:
            continue
        tag = _symmetric_band(
            nutrient, float(actual), target_value, unit, _MACRO_TOLERANCE
        )
        if tag is not None:
            tags.append(tag)

    # AGS plafond : excess seul si > 12 % AET du repas.
    actual_ags = meal_macros.get("saturated_fat_g")
    if actual_ags is not None:
        ags_target = (
            targets.calories * config.RNP_AGS_PERCENT_OF_AET_MAX / _KCAL_PER_FAT_GRAM
        )
        ags_tag = _ceiling_only(
            Nutrient.saturated_fat_g, float(actual_ags), ags_target, "g"
        )
        if ags_tag is not None:
            tags.append(ags_tag)

    # Fibres deficit : RNP 30 g/j ANSES, ramene a la part du repas. Quota
    # repas si meal_type connu, sinon un quart du RNP (cf. nutrition_engine).
    actual_fibers = meal_macros.get("fibers_g")
    if actual_fibers is None:
        # Compatibilite avec les payloads existants utilisant la cle 'fiber_g'
        # (cf. nutrition_lookup et meal_analyses.macros stockes en JSONB).
        actual_fibers = meal_macros.get("fiber_g")
    if actual_fibers is not None:
        fibers_target = config.RNP_FIBER_G_PER_DAY * meal_quota(meal_type)
        fibers_tag = _floor_only(
            Nutrient.fibers_g, float(actual_fibers), fibers_target, "g"
        )
        if fibers_tag is not None:
            tags.append(fibers_tag)

    return tags


_NUTRIENT_LABEL: dict[Nutrient, str] = {
    Nutrient.calories: "calories",
    Nutrient.protein_g: "proteines",
    Nutrient.carbs_g: "glucides",
    Nutrient.fat_g: "lipides",
    Nutrient.fibers_g: "fibres",
    Nutrient.saturated_fat_g: "acides gras satures",
}

_EXCESS_PHRASES: dict[Nutrient, str] = {
    Nutrient.calories: "Apport calorique eleve",
    Nutrient.protein_g: "Trop de proteines",
    Nutrient.carbs_g: "Trop de glucides",
    Nutrient.fat_g: "Trop de lipides",
    Nutrient.fibers_g: "Trop de fibres",
    Nutrient.saturated_fat_g: "Trop d'acides gras satures",
}

_DEFICIT_PHRASES: dict[Nutrient, str] = {
    Nutrient.calories: "Apport calorique faible",
    Nutrient.protein_g: "Faible en proteines",
    Nutrient.carbs_g: "Faible en glucides",
    Nutrient.fat_g: "Faible en lipides",
    Nutrient.fibers_g: "Manque de fibres",
    Nutrient.saturated_fat_g: "Faible en acides gras satures",
}


def imbalance_to_text(tag: ImbalanceTag) -> str:
    """Convertit un tag en phrase francaise deterministe (sans appel LLM).

    Format : "<phrase> : <actual><unit> (cible <target><unit>, ecart <signe><pct>%)".
    Le pourcentage est arrondi a l'unite ; la cible et l'apport reel a 1 chiffre
    pour les grammes et a l'unite pour les kcal.
    """
    if tag.status is ImbalanceStatus.excess:
        head = _EXCESS_PHRASES[tag.nutrient]
    elif tag.status is ImbalanceStatus.deficit:
        head = _DEFICIT_PHRASES[tag.nutrient]
    else:
        # ok : pas de phrase d'alerte. On retourne quand meme un texte neutre
        # pour ne pas casser un caller qui itererait sur tous les tags.
        return f"{_NUTRIENT_LABEL[tag.nutrient]} dans la cible."

    actual = _format_value(tag.actual_value, tag.unit)
    target = _format_value(tag.target_value, tag.unit)
    sign = "+" if tag.delta_pct >= 0 else ""
    pct = round(tag.delta_pct * 100)
    return (
        f"{head} : {actual} {tag.unit} "
        f"(cible {target} {tag.unit}, ecart {sign}{pct}%)"
    )


def _format_value(value: float, unit: str) -> str:
    # kcal : entier ; grammes : 1 chiffre apres la virgule.
    if unit == "kcal":
        return f"{value:.0f}"
    return f"{value:.1f}"


def _ceiling_only(
    nutrient: Nutrient, actual: float, target: float, unit: str
) -> ImbalanceTag | None:
    """Asymetrique : on signale uniquement un excess > +20 %. Pas de deficit."""
    if target <= 0:
        return None
    delta = (actual - target) / target
    if delta <= _MACRO_TOLERANCE:
        return None
    return ImbalanceTag(
        nutrient=nutrient,
        status=ImbalanceStatus.excess,
        delta_pct=delta,
        target_value=target,
        actual_value=actual,
        unit=unit,
    )


def _floor_only(
    nutrient: Nutrient, actual: float, target: float, unit: str
) -> ImbalanceTag | None:
    """Asymetrique : on signale uniquement un deficit > 20 %. Pas d'excess."""
    if target <= 0:
        return None
    delta = (actual - target) / target
    if delta >= -_MACRO_TOLERANCE:
        return None
    return ImbalanceTag(
        nutrient=nutrient,
        status=ImbalanceStatus.deficit,
        delta_pct=delta,
        target_value=target,
        actual_value=actual,
        unit=unit,
    )


def _symmetric_band(
    nutrient: Nutrient,
    actual: float,
    target: float,
    unit: str,
    tolerance: float,
) -> ImbalanceTag | None:
    """Bande symetrique autour de la cible : excess ou deficit au-dela de tolerance."""
    if target <= 0:
        return None
    delta = (actual - target) / target
    if delta > tolerance:
        return ImbalanceTag(
            nutrient=nutrient,
            status=ImbalanceStatus.excess,
            delta_pct=delta,
            target_value=target,
            actual_value=actual,
            unit=unit,
        )
    if delta < -tolerance:
        return ImbalanceTag(
            nutrient=nutrient,
            status=ImbalanceStatus.deficit,
            delta_pct=delta,
            target_value=target,
            actual_value=actual,
            unit=unit,
        )
    return None
