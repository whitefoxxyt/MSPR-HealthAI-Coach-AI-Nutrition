from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from app.models.schemas import FallbackMealPlan, MealDay


class ViolationType(str, Enum):
    allergy = "allergy"
    budget = "budget"
    diet = "diet"


@dataclass(frozen=True)
class ConstraintSpec:
    """Contraintes utilisateur : allergies + budget journalier + regime."""

    allergies: list[str] = field(default_factory=list)
    max_daily_budget_eur: float | None = None
    diet_type: str | None = None


# Heuristique simple : liste non exhaustive d'ingredients incompatibles avec
# chaque regime alimentaire courant. Le matching reste a frontiere de mot.
_DIET_BANNED: dict[str, frozenset[str]] = {
    "vegan": frozenset(
        {
            "viande",
            "boeuf",
            "poulet",
            "porc",
            "agneau",
            "veau",
            "dinde",
            "canard",
            "jambon",
            "lardon",
            "saucisse",
            "bacon",
            "poisson",
            "saumon",
            "thon",
            "crevette",
            "crevettes",
            "fruit de mer",
            "fruits de mer",
            "lait",
            "laitiere",
            "fromage",
            "yaourt",
            "beurre",
            "creme",
            "miel",
            "oeuf",
        }
    ),
    "vegetarien": frozenset(
        {
            "viande",
            "boeuf",
            "poulet",
            "porc",
            "agneau",
            "veau",
            "dinde",
            "canard",
            "jambon",
            "lardon",
            "saucisse",
            "bacon",
            "poisson",
            "saumon",
            "thon",
            "crevette",
            "crevettes",
            "fruit de mer",
            "fruits de mer",
        }
    ),
    "sans_gluten": frozenset(
        {
            "ble",
            "farine de ble",
            "pain",
            "pates",
            "couscous",
            "boulgour",
            "seigle",
            "orge",
            "avoine",
        }
    ),
}


@dataclass(frozen=True)
class ConstraintViolation:
    """Violation localisee : type + ou (jour, repas) + valeur fautive + message."""

    type: ViolationType
    day: int
    meal_index: int | None
    ingredient_or_amount: str | float
    message: str


def _normalize(text_value: str) -> str:
    """Casefold + strip des accents pour matcher 'lait' contre 'Lait écrémé'."""
    decomposed = unicodedata.normalize("NFD", text_value)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.casefold()


def _compile_term(term: str) -> re.Pattern[str]:
    """Pattern a frontiere de mot, evite 'lait' dans 'laitue' ou 'oeuf' dans 'boeuf'."""
    return re.compile(rf"\b{re.escape(_normalize(term))}\b")


def validate(
    plan: FallbackMealPlan, constraints: ConstraintSpec
) -> list[ConstraintViolation]:
    """Retourne toutes les violations du plan vis-a-vis des contraintes."""
    allergen_patterns = [
        (a, _compile_term(a)) for a in constraints.allergies if a.strip()
    ]
    banned = _DIET_BANNED.get(constraints.diet_type or "", frozenset())
    banned_patterns = [(t, _compile_term(t)) for t in banned]
    diet_label = constraints.diet_type
    budget_max = constraints.max_daily_budget_eur

    violations: list[ConstraintViolation] = []
    for day in plan.days:
        for meal_idx, meal in enumerate(day.meals):
            for ing in meal.ingredients:
                normalized = _normalize(ing)
                violations.extend(
                    _allergen_violations(
                        ing, normalized, allergen_patterns, day.day, meal_idx
                    )
                )
                violations.extend(
                    _diet_violations(
                        ing, normalized, banned_patterns, diet_label, day.day, meal_idx
                    )
                )
        if budget_max is not None:
            violations.extend(_budget_violation(day, budget_max))
    return violations


def _allergen_violations(
    ingredient: str,
    normalized_ingredient: str,
    allergen_patterns: list[tuple[str, re.Pattern[str]]],
    day_num: int,
    meal_idx: int,
) -> list[ConstraintViolation]:
    return [
        ConstraintViolation(
            type=ViolationType.allergy,
            day=day_num,
            meal_index=meal_idx,
            ingredient_or_amount=ingredient,
            message=f"Allergene {a!r} present dans {ingredient!r}.",
        )
        for a, pattern in allergen_patterns
        if pattern.search(normalized_ingredient)
    ]


def _diet_violations(
    ingredient: str,
    normalized_ingredient: str,
    banned_patterns: list[tuple[str, re.Pattern[str]]],
    diet_type: str | None,
    day_num: int,
    meal_idx: int,
) -> list[ConstraintViolation]:
    return [
        ConstraintViolation(
            type=ViolationType.diet,
            day=day_num,
            meal_index=meal_idx,
            ingredient_or_amount=ingredient,
            message=(
                f"Ingredient {ingredient!r} incompatible avec le regime "
                f"{diet_type!r} (terme banni : {term!r})."
            ),
        )
        for term, pattern in banned_patterns
        if pattern.search(normalized_ingredient)
    ]


def _budget_violation(day: MealDay, budget_max: float) -> list[ConstraintViolation]:
    day_total = sum(meal.est_budget_eur for meal in day.meals)
    if day_total <= budget_max:
        return []
    return [
        ConstraintViolation(
            type=ViolationType.budget,
            day=day.day,
            meal_index=None,
            ingredient_or_amount=day_total,
            message=(
                f"Budget journalier {day_total:.2f} EUR "
                f"depasse la limite {budget_max:.2f} EUR."
            ),
        )
    ]
