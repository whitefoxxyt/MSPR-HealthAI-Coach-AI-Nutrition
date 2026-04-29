from __future__ import annotations

from app.db.models import NutritionGoal
from app.models.schemas import Imbalance

# Detection des desequilibres macros par rapport aux objectifs journaliers.
#
# Renvoie une liste de (Imbalance, message) ordonnee : calories -> proteines ->
# glucides -> lipides. Cet ordre stable est utilise pour le hash de cache LLM
# (issue NUT-11).

_CALORIES_RATIO = 0.6
_PROTEIN_RATIO = 0.2
_CARBS_RATIO = 0.7
_FAT_RATIO = 0.7


def detect_imbalances(
    macros: dict, goal: NutritionGoal | None
) -> list[tuple[Imbalance, str]]:
    if goal is None or not macros:
        return []
    issues: list[tuple[Imbalance, str]] = []

    if goal.calories_target and macros.get("calories") is not None:
        calories = macros["calories"]
        ratio = calories / float(goal.calories_target)
        if ratio > _CALORIES_RATIO:
            issues.append(
                (
                    Imbalance.calories_high,
                    f"Apport calorique eleve : {calories:.0f} kcal "
                    f"({ratio:.0%} de l'objectif journalier de {goal.calories_target} kcal)",
                )
            )

    if goal.protein_g and macros.get("protein_g") is not None:
        protein = macros["protein_g"]
        if protein < float(goal.protein_g) * _PROTEIN_RATIO:
            issues.append(
                (
                    Imbalance.protein_low,
                    f"Faible en proteines : {protein:.1f}g "
                    f"(objectif journalier : {goal.protein_g}g)",
                )
            )

    if goal.carbs_g and macros.get("carbs_g") is not None:
        carbs = macros["carbs_g"]
        if carbs > float(goal.carbs_g) * _CARBS_RATIO:
            issues.append(
                (
                    Imbalance.carbs_high,
                    f"Eleve en glucides : {carbs:.1f}g "
                    f"(objectif journalier : {goal.carbs_g}g)",
                )
            )

    if goal.fat_g and macros.get("fat_g") is not None:
        fat = macros["fat_g"]
        if fat > float(goal.fat_g) * _FAT_RATIO:
            issues.append(
                (
                    Imbalance.fat_high,
                    f"Eleve en lipides : {fat:.1f}g "
                    f"(objectif journalier : {goal.fat_g}g)",
                )
            )

    return issues
