from __future__ import annotations

import csv
import math
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from app.models.schemas import FallbackMealPlan

# Heuristique simple pour le respect du regime alimentaire (vegan, vegetarien,
# sans_gluten). Liste non exhaustive : c'est une eval, pas un nutritionniste.
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
class GenerationOutcome:
    """Resultat d'une generation Ollama, du point de vue de l'eval."""

    json_valid_first_try: bool
    used_fallback: bool
    latency_ms: float


@dataclass(frozen=True)
class ConstraintCheck:
    """Trois drapeaux : allergies absentes, budget respecte, regime respecte."""

    allergies_absent: bool
    budget_respected: bool
    diet_respected: bool

    def all_satisfied(self) -> bool:
        return self.allergies_absent and self.budget_respected and self.diet_respected


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """Renvoie p50/p95/max en millisecondes (methode nearest-rank).

    Convention : si la liste est vide, renvoie des zeros plutot que de lever,
    car une eval ou tous les appels echouent reste un resultat valide.
    """
    if not latencies_ms:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    sorted_latencies = sorted(latencies_ms)
    n = len(sorted_latencies)
    return {
        "p50_ms": sorted_latencies[max(0, math.ceil(0.50 * n) - 1)],
        "p95_ms": sorted_latencies[max(0, math.ceil(0.95 * n) - 1)],
        "max_ms": sorted_latencies[-1],
    }


def json_validity_rate(outcomes: list[GenerationOutcome]) -> float:
    """Ratio des generations dont le JSON est valide au premier essai."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.json_valid_first_try) / len(outcomes)


def fallback_rate(outcomes: list[GenerationOutcome]) -> float:
    """Ratio des generations qui ont bascule sur le fallback statique."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.used_fallback) / len(outcomes)


def constraint_satisfaction_rate(checks: list[ConstraintCheck]) -> float:
    """Ratio des plans qui respectent simultanement allergies + budget + regime."""
    if not checks:
        return 0.0
    return sum(1 for c in checks if c.all_satisfied()) / len(checks)


@dataclass(frozen=True)
class ConstraintSpec:
    """Contraintes a verifier sur un plan : allergies + budget + regime."""

    allergies: list[str] = field(default_factory=list)
    max_daily_budget_eur: float | None = None
    diet_type: str | None = None


def _normalize(text_value: str) -> str:
    decomposed = unicodedata.normalize("NFD", text_value)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return stripped.casefold()


def _ingredient_contains_term(ingredient: str, term: str) -> bool:
    """Match a frontiere de mot, comme llm_client._enforce_allergy_absence."""
    pattern = re.compile(rf"\b{re.escape(_normalize(term))}\b")
    return bool(pattern.search(_normalize(ingredient)))


def check_plan_constraints(
    plan: FallbackMealPlan,
    spec: ConstraintSpec,
) -> ConstraintCheck:
    """Verifie qu'un plan respecte allergies + budget + regime."""
    allergies_absent = True
    if spec.allergies:
        for day in plan.days:
            for meal in day.meals:
                for ing in meal.ingredients:
                    for allergen in spec.allergies:
                        if _ingredient_contains_term(ing, allergen):
                            allergies_absent = False
                            break

    budget_respected = True
    if spec.max_daily_budget_eur is not None:
        for day in plan.days:
            day_cost = sum(meal.est_budget_eur for meal in day.meals)
            if day_cost > spec.max_daily_budget_eur:
                budget_respected = False
                break

    diet_respected = True
    banned = _DIET_BANNED.get(spec.diet_type or "", frozenset())
    if banned:
        for day in plan.days:
            for meal in day.meals:
                for ing in meal.ingredients:
                    for term in banned:
                        if _ingredient_contains_term(ing, term):
                            diet_respected = False
                            break

    return ConstraintCheck(
        allergies_absent=allergies_absent,
        budget_respected=budget_respected,
        diet_respected=diet_respected,
    )


@dataclass(frozen=True)
class HitlSummary:
    """Moyennes des notations humaines (1 a 5) sur 3 dimensions."""

    n_ratings: int
    mean_nutrition: float
    mean_originalite: float
    mean_coherence: float


def load_hitl_ratings(csv_path: Path) -> HitlSummary:
    """Lit un CSV plan_id,nutrition,originalite,coherence -> moyennes."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, str]] = [row for row in reader]
    n = len(rows)
    if n == 0:
        return HitlSummary(0, 0.0, 0.0, 0.0)
    nutrition = sum(float(r["nutrition"]) for r in rows) / n
    originalite = sum(float(r["originalite"]) for r in rows) / n
    coherence = sum(float(r["coherence"]) for r in rows) / n
    return HitlSummary(
        n_ratings=n,
        mean_nutrition=nutrition,
        mean_originalite=originalite,
        mean_coherence=coherence,
    )
