"""
Runner LLM : lance N generations Ollama (gemma3:4b) et calcule les 5 metriques
exigees : validite JSON 1er essai, latence p50/p95/max, taux de fallback,
respect simultanee allergies+budget+regime, et resume HITL si CSV present.

Necessite Ollama accessible (settings.ollama_host) et le modele gemma3:4b pull.
"""

from __future__ import annotations

import json
import logging
import random
import time
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from app.config import settings
from app.models.schemas import (
    DietType,
    FallbackMealPlan,
    HealthGoal,
    PlanInputs,
)
from scripts.eval.llm_metrics import (
    ConstraintCheck,
    ConstraintSpec,
    GenerationOutcome,
    check_plan_constraints,
    constraint_satisfaction_rate,
    fallback_rate,
    json_validity_rate,
    latency_percentiles,
    load_hitl_ratings,
)
from scripts.eval.plotting import save_latency_distribution_png

logger = logging.getLogger(__name__)

_OLLAMA_MODEL = "gemma3:4b"
_OLLAMA_TIMEOUT_S = 180.0


_PROMPT_TEMPLATE = (
    "Tu es un nutritionniste. Genere un plan repas JSON pour {duration} jours.\n"
    "Objectif : {objective}.\n"
    "Regime : {diet}.\n"
    "Allergies a eviter : {allergies}.\n"
    "Cible calorique journaliere : {calories}.\n"
    "Pour chaque repas : name, macros (calories, protein_g, carbs_g, fat_g),\n"
    "ingredients (liste), est_budget_eur, prep_time_min. Mets fallback=false.\n"
    "Reponds uniquement par un JSON conforme au schema fourni."
)


async def run_llm_eval(
    n_generations: int,
    n_constraint_plans: int,
    hitl_csv: Path,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Renvoie un payload llm pour metrics.json."""
    logger.info("llm : %d generations en conditions normales", n_generations)
    outcomes = await _run_generations(n_generations, with_constraints=False, seed=seed)

    constraint_outcomes, constraint_checks = await _run_constraint_eval(
        n_constraint_plans, seed=seed
    )
    outcomes.extend(constraint_outcomes)

    successful_latencies = [o.latency_ms for o in outcomes if not o.used_fallback]
    latency_png = output_dir / "llm_latency_distribution.png"
    if successful_latencies:
        save_latency_distribution_png(successful_latencies, latency_png)

    payload: dict[str, Any] = {
        "n_generations": len(outcomes),
        "json_validity_rate": json_validity_rate(outcomes),
        "fallback_rate": fallback_rate(outcomes),
        "latency": latency_percentiles(successful_latencies),
        "constraint_satisfaction_rate": constraint_satisfaction_rate(constraint_checks),
        "constraint_n_plans": len(constraint_checks),
        "latency_distribution_png": str(latency_png) if successful_latencies else None,
    }

    if hitl_csv.exists():
        summary = load_hitl_ratings(hitl_csv)
        payload["hitl"] = {
            "n_ratings": summary.n_ratings,
            "mean_nutrition": summary.mean_nutrition,
            "mean_originalite": summary.mean_originalite,
            "mean_coherence": summary.mean_coherence,
        }
    else:
        logger.warning(
            "hitl : %s introuvable, evaluation qualitative non incluse. "
            "Notez 20 plans dans le template puis relancer.",
            hitl_csv,
        )

    return payload


# --- internes : generations brutes via Ollama --------------------------------


async def _run_generations(
    n: int,
    with_constraints: bool,
    seed: int,
) -> list[GenerationOutcome]:
    """Genere n plans 'normaux' (pas de contraintes), mesure validite + latence."""
    outcomes: list[GenerationOutcome] = []
    rng = random.Random(seed)
    for i in range(n):
        inputs = _random_inputs(rng, i, with_constraints=with_constraints)
        outcome, _ = await _call_ollama_for_eval(inputs)
        outcomes.append(outcome)
    return outcomes


async def _run_constraint_eval(
    n: int,
    seed: int,
) -> tuple[list[GenerationOutcome], list[ConstraintCheck]]:
    """Genere n plans contraints (allergies + budget + regime) et evalue le respect."""
    outcomes: list[GenerationOutcome] = []
    checks: list[ConstraintCheck] = []
    rng = random.Random(seed + 1)  # Decorele du seed des generations 'normales'
    for i in range(n):
        inputs = _random_inputs(rng, i, with_constraints=True)
        spec = ConstraintSpec(
            allergies=list(inputs.allergies),
            max_daily_budget_eur=float(inputs.budget_per_day)
            if inputs.budget_per_day
            else None,
            diet_type=inputs.diet_type,
        )
        outcome, plan = await _call_ollama_for_eval(inputs)
        outcomes.append(outcome)
        if plan is not None and not outcome.used_fallback:
            checks.append(check_plan_constraints(plan, spec))
    return outcomes, checks


async def _call_ollama_for_eval(
    inputs: PlanInputs,
) -> tuple[GenerationOutcome, FallbackMealPlan | None]:
    """Appel Ollama brut. On evite generate_plan (DB requise) et on instrumente
    nous-memes la validite JSON 1er essai et la latence wallclock."""
    prompt = _build_prompt(inputs)
    schema = FallbackMealPlan.model_json_schema()

    start = time.monotonic()
    raw_response: str | None = None
    try:
        async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_S) as client:
            resp = await client.post(
                f"{settings.ollama_host}/api/generate",
                json={
                    "model": _OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "format": schema,
                },
            )
            resp.raise_for_status()
            raw_response = resp.json().get("response", "")
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.warning("eval ollama HTTPError : %s", exc)
        return (
            GenerationOutcome(
                json_valid_first_try=False,
                used_fallback=True,
                latency_ms=latency_ms,
            ),
            None,
        )

    latency_ms = (time.monotonic() - start) * 1000
    plan, json_ok = _parse_response(raw_response or "")
    return (
        GenerationOutcome(
            json_valid_first_try=json_ok,
            used_fallback=plan is None,
            latency_ms=latency_ms,
        ),
        plan,
    )


def _parse_response(raw: str) -> tuple[FallbackMealPlan | None, bool]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None, False
    if isinstance(parsed, dict):
        parsed.setdefault("fallback", False)
    try:
        return FallbackMealPlan.model_validate(parsed), True
    except ValidationError:
        return None, False


def _build_prompt(inputs: PlanInputs) -> str:
    return _PROMPT_TEMPLATE.format(
        duration=inputs.duration_days,
        objective=inputs.objective,
        diet=inputs.diet_type or "aucun",
        allergies=", ".join(sorted(inputs.allergies)) if inputs.allergies else "aucune",
        calories=inputs.calories_target if inputs.calories_target else "non specifiee",
    )


# Echantillonnage de PlanInputs reproductible
_GOALS = [g.value for g in HealthGoal]
_DIETS = [d.value for d in DietType]
_ALLERGEN_POOL = ["arachide", "lait", "gluten", "oeuf", "soja", "fruits a coque"]


def _random_inputs(rng: random.Random, idx: int, with_constraints: bool) -> PlanInputs:
    objective = rng.choice(_GOALS)
    if with_constraints:
        diet = rng.choice(_DIETS)
        allergies = rng.sample(_ALLERGEN_POOL, k=rng.randint(1, 2))
        budget = Decimal(str(round(rng.uniform(8.0, 20.0), 2)))
    else:
        diet = rng.choice(_DIETS)
        allergies = []
        budget = None
    return PlanInputs(
        user_id=1_000_000 + idx,
        objective=objective,
        duration_days=1,
        diet_type=diet,
        allergies=allergies,
        budget_per_day=budget,
    )
