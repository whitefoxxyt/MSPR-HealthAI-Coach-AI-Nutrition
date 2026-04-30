from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from app.models.schemas import FallbackMealPlan
from app.services.constraint_validator import (
    ConstraintSpec,
    ViolationType,
    validate as validate_constraints,
)

__all__ = [
    "ConstraintCheck",
    "ConstraintSpec",
    "GenerationOutcome",
    "HitlSummary",
    "check_plan_constraints",
    "constraint_satisfaction_rate",
    "fallback_rate",
    "json_validity_rate",
    "latency_percentiles",
    "load_hitl_ratings",
]


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


def check_plan_constraints(
    plan: FallbackMealPlan,
    spec: ConstraintSpec,
) -> ConstraintCheck:
    """Verifie qu'un plan respecte allergies + budget + regime.

    Adaptateur autour de constraint_validator.validate : agrege les violations
    granulaires en trois drapeaux booleens pour l'eval.
    """
    types = {v.type for v in validate_constraints(plan, spec)}
    return ConstraintCheck(
        allergies_absent=ViolationType.allergy not in types,
        budget_respected=ViolationType.budget not in types,
        diet_respected=ViolationType.diet not in types,
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
