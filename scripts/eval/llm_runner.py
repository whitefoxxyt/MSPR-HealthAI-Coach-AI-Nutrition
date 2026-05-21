"""
Runner LLM : compare deux niveaux de pipeline sur les memes inputs aleatoires.

  - llm.naive    : LLM nu (sans DeCRIM-light), tel qu'utilise avant slice 7.
                   Mesure validite JSON, latence, fallback, respect simultanee
                   allergies+budget+regime, et HITL.
  - llm.pipeline : pipeline complet (slice 7) generate_plan + decrim_retry_orchestrator.
                   Mesure compliance_status (full / partial_budget / static_fallback /
                   abandoned_503), latence, retries LLM, respect par contrainte.

Backend selectionne par la var d'env LLM_BACKEND (slice 5 #75) : 'ollama' (defaut,
gemma3:4b local) ou 'mistral' (mistral-small managed via API).

Necessite :
  - Backend cible accessible : Ollama (settings.ollama_host + modele pull) ou
    Mistral (MISTRAL_API_KEY)
  - PostgreSQL accessible (settings.database_url) avec migrations V11+ jouees
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import time
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
from pydantic import ValidationError

from app.models.schemas import (
    DietType,
    FallbackMealPlan,
    HealthGoal,
    PlanInputs,
)
from app.services.llm_provider import LLMProvider, get_provider
from scripts.eval.llm_metrics import (
    ConstraintCheck,
    ConstraintSpec,
    GenerationOutcome,
    PipelineOutcome,
    by_constraint_satisfaction,
    check_plan_constraints,
    compliance_status_breakdown,
    constraint_satisfaction_rate,
    fallback_rate,
    json_validity_rate,
    latency_percentiles,
    load_hitl_ratings,
    retry_count_distribution,
)
from scripts.eval.plotting import save_latency_distribution_png

logger = logging.getLogger(__name__)


def _resolve_backend() -> str:
    """Lit LLM_BACKEND env var (defaut ollama, retrocompat slice 5 #75)."""
    return os.environ.get("LLM_BACKEND", "ollama").lower()


_BACKEND_ALIASES = {"ollama": "gemma"}


def _run_key(backend: str, n: int) -> str:
    """Cle metrics.json pour un run donne : alias + nombre de generations.

    Convention slice 5 (#75) : 'ollama' s'affiche 'gemma_n<N>' (l'utilisateur
    parle du modele, pas du serveur d'inference). Mistral garde son nom.
    """
    alias = _BACKEND_ALIASES.get(backend.lower(), backend.lower())
    return f"{alias}_n{n}"


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
    """Renvoie un payload llm keye par run (slice 5 #75).

    Forme : `{<run_key>: {backend, naive, pipeline}}` ou `<run_key>` est par
    exemple `gemma_n20` ou `mistral_n100`. Le backend est lu depuis l'env var
    `LLM_BACKEND` (defaut ollama). Permet d'empiler 3 runs comparatifs dans le
    meme `docs/metrics.json` (cf. _persist qui merge la section llm).
    """
    backend = _resolve_backend()
    provider = get_provider(backend)

    naive_payload = await _run_naive_eval(
        n_generations=n_generations,
        n_constraint_plans=n_constraint_plans,
        hitl_csv=hitl_csv,
        seed=seed,
        output_dir=output_dir,
        provider=provider,
    )

    pipeline_payload = await _run_pipeline_eval(
        n=n_constraint_plans,
        seed=seed,
        primary_backend=backend,
    )

    return {
        _run_key(backend, n_constraint_plans): {
            "backend": backend,
            "naive": naive_payload,
            "pipeline": pipeline_payload,
        }
    }


# Niveau naive : LLM nu (gemma3:4b sans DeCRIM-light, sans cache, sans persistance).


async def _run_naive_eval(
    n_generations: int,
    n_constraint_plans: int,
    hitl_csv: Path,
    seed: int,
    output_dir: Path,
    provider: LLMProvider,
) -> dict[str, Any]:
    """Eval du LLM nu : pas de retry sur contraintes, pas de fallback hierarchique."""
    logger.info("llm.naive : %d generations en conditions normales", n_generations)
    outcomes = await _run_generations(
        n_generations, with_constraints=False, seed=seed, provider=provider
    )

    constraint_outcomes, constraint_checks = await _run_constraint_eval(
        n_constraint_plans, seed=seed, provider=provider
    )
    outcomes.extend(constraint_outcomes)

    successful_latencies = [o.latency_ms for o in outcomes if not o.used_fallback]
    latency_png = output_dir / "llm_latency_distribution.png"
    if successful_latencies:
        save_latency_distribution_png(successful_latencies, latency_png)

    latency = latency_percentiles(successful_latencies)
    payload: dict[str, Any] = {
        "n_generations": len(outcomes),
        "json_validity_rate": json_validity_rate(outcomes),
        "fallback_rate": fallback_rate(outcomes),
        "latency_p50_ms": latency["p50_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "latency_max_ms": latency["max_ms"],
        "constraint_satisfaction": constraint_satisfaction_rate(constraint_checks),
        "constraint_n_plans": len(constraint_checks),
        "by_constraint": by_constraint_satisfaction(constraint_checks),
        "latency_distribution_png": str(latency_png) if successful_latencies else None,
        # Slice 5 (#75) : conserve les latences brutes pour le PNG comparatif
        # multi-backend rendu apres tous les runs (cf. eval_metrics._persist).
        "latencies_ms_raw": successful_latencies,
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


# Niveau pipeline : generate_plan complet (DeCRIM-light + cache bypass + persistance).


async def _run_pipeline_eval(n: int, seed: int, primary_backend: str) -> dict[str, Any]:
    """Genere n plans via le pipeline complet, memes inputs que _run_constraint_eval.

    Compte les compliance_status, mesure latence + retries LLM, et calcule
    le respect par contrainte (allergies / budget / diet) sur les plans rendus.
    `primary_backend` selectionne ollama vs mistral (slice 5 #75).
    """
    # Imports tardifs : ces modules touchent a la BDD et au LLM, on ne les
    # charge que dans le pipeline eval (le naive eval reste DB-less).
    from app.db.session import SessionLocal
    from app.services.decrim_retry_orchestrator import InfeasibleConstraintsError
    from app.services.fallback_loader import load_fallback_plan
    from app.services.llm_client import generate_plan

    logger.info(
        "llm.pipeline : %d generations via generate_plan (bypass_cache, backend=%s)",
        n,
        primary_backend,
    )

    pipeline_outcomes: list[PipelineOutcome] = []
    rng = random.Random(seed + 1)  # Meme seed que _run_constraint_eval -> memes inputs.

    for i in range(n):
        inputs = _random_inputs(rng, i, with_constraints=True)
        spec = _spec_from_inputs(inputs)

        async with _instrument_chain_calls() as call_counter:
            start = time.monotonic()
            try:
                with _ephemeral_session(SessionLocal) as db:
                    plan, status, _warnings = await generate_plan(
                        inputs,
                        db,
                        bypass_cache=True,
                        fallback_loader=load_fallback_plan,
                        primary_backend=primary_backend,
                    )
                latency_ms = (time.monotonic() - start) * 1000
                check = check_plan_constraints(plan, spec)
                pipeline_outcomes.append(
                    PipelineOutcome(
                        compliance_status=status.value,
                        ollama_calls=call_counter["calls"],
                        latency_ms=latency_ms,
                        constraint_check=check,
                    )
                )
            except InfeasibleConstraintsError:
                latency_ms = (time.monotonic() - start) * 1000
                pipeline_outcomes.append(
                    PipelineOutcome(
                        compliance_status="abandoned_503",
                        ollama_calls=call_counter["calls"],
                        latency_ms=latency_ms,
                        constraint_check=None,
                    )
                )

    breakdown = compliance_status_breakdown(pipeline_outcomes)
    latencies = [o.latency_ms for o in pipeline_outcomes]
    latency = latency_percentiles(latencies)
    checks = [o.constraint_check for o in pipeline_outcomes if o.constraint_check]

    return {
        "n_generations": n,
        "constraint_satisfaction": breakdown["full"],
        "partial_compliance": breakdown["partial_budget"],
        "static_fallback": breakdown["static_fallback"],
        "abandoned_503": breakdown["abandoned_503"],
        "by_constraint": by_constraint_satisfaction(checks),
        "latency_p50_ms": latency["p50_ms"],
        "latency_p95_ms": latency["p95_ms"],
        "retry_count_distribution": retry_count_distribution(pipeline_outcomes),
    }


class _CountingChain:
    """Wrapper API-compatible FallbackChain qui compte les appels generate().

    Slice 5 (#75) remplace l'ancien monkeypatch sur `_ollama_generate` (purge au
    slice 2 au profit du chain multi-provider). On compte au niveau du chain car
    c'est l'unite d'inference du nouveau pipeline.
    """

    def __init__(self, inner: Any, counter: dict[str, int]) -> None:
        self._inner = inner
        self._counter = counter

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None,
        primary_backend: str,
    ) -> tuple[str, str]:
        self._counter["calls"] += 1
        return await self._inner.generate(prompt, schema, primary_backend)


@contextlib.asynccontextmanager
async def _instrument_chain_calls() -> AsyncIterator[dict[str, int]]:
    """Patche `build_default_chain` cote decrim_retry_orchestrator pour produire
    un chain comptant les appels generate(). Restaure l'original a la sortie.

    Slice 5 (#75) remplace `_count_ollama_calls` casse depuis slice 2 (la fct
    `_ollama_generate` qu'il patchait n'existe plus).
    """
    from app.services import decrim_retry_orchestrator as orch

    counter: dict[str, int] = {"calls": 0}
    original_builder = orch.build_default_chain

    def patched_builder() -> _CountingChain:
        return _CountingChain(original_builder(), counter)

    orch.build_default_chain = patched_builder
    try:
        yield counter
    finally:
        orch.build_default_chain = original_builder


@contextlib.contextmanager
def _ephemeral_session(session_factory: Any) -> Any:
    """Session SQLAlchemy ouverte/fermee proprement, rollback en sortie.

    On rollback en fin de boucle : on ne veut pas polluer la BDD avec les
    plans d'eval (les resultats sont aggreges en memoire et exportes en JSON).
    """
    db = session_factory()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


def _spec_from_inputs(inputs: PlanInputs) -> ConstraintSpec:
    return ConstraintSpec(
        allergies=list(inputs.allergies),
        max_daily_budget_eur=float(inputs.budget_per_day)
        if inputs.budget_per_day
        else None,
        diet_type=inputs.diet_type,
    )


# --- internes : generations brutes via le provider LLM (niveau naive) --------


async def _run_generations(
    n: int,
    with_constraints: bool,
    seed: int,
    provider: LLMProvider,
) -> list[GenerationOutcome]:
    """Genere n plans 'normaux' (pas de contraintes), mesure validite + latence."""
    outcomes: list[GenerationOutcome] = []
    rng = random.Random(seed)
    for i in range(n):
        inputs = _random_inputs(rng, i, with_constraints=with_constraints)
        outcome, _ = await _call_for_eval(provider, inputs)
        outcomes.append(outcome)
    return outcomes


async def _run_constraint_eval(
    n: int,
    seed: int,
    provider: LLMProvider,
) -> tuple[list[GenerationOutcome], list[ConstraintCheck]]:
    """Genere n plans contraints (allergies + budget + regime) et evalue le respect."""
    outcomes: list[GenerationOutcome] = []
    checks: list[ConstraintCheck] = []
    rng = random.Random(seed + 1)  # Decorele du seed des generations 'normales'
    for i in range(n):
        inputs = _random_inputs(rng, i, with_constraints=True)
        spec = _spec_from_inputs(inputs)
        outcome, plan = await _call_for_eval(provider, inputs)
        outcomes.append(outcome)
        if plan is not None and not outcome.used_fallback:
            checks.append(check_plan_constraints(plan, spec))
    return outcomes, checks


async def _call_for_eval(
    provider: LLMProvider,
    inputs: PlanInputs,
) -> tuple[GenerationOutcome, FallbackMealPlan | None]:
    """Appel LLM brut via le provider injecte (slice 5 #75).

    On evite generate_plan (DB requise) et on instrumente nous-memes la
    validite JSON 1er essai et la latence wallclock. Le provider gere les
    specificites HTTP (format Ollama, response_format Mistral strict).
    """
    prompt = _build_prompt(inputs)
    schema = FallbackMealPlan.model_json_schema()

    start = time.monotonic()
    try:
        raw_response = await provider.generate(prompt, schema)
    except httpx.HTTPError as exc:
        latency_ms = (time.monotonic() - start) * 1000
        logger.warning("eval llm HTTPError : %s", exc)
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
