from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.models.schemas import (
    FallbackMealPlan,
    HealthGoal,
    Imbalance,
    PlanInputs,
    RecommendationContext,
)
from app.services.constraint_validator import (
    ConstraintSpec,
    ViolationType,
    validate as validate_constraints,
)

T = TypeVar("T")

_LOGGER = logging.getLogger(__name__)

# Limite la charge CPU d'Ollama : au plus 2 inferences simultanees, peu importe
# le nombre de coroutines qui appellent generate_plan.
_OLLAMA_SEMAPHORE = asyncio.Semaphore(2)

_OLLAMA_TIMEOUT_S = 30.0
_MAX_ATTEMPTS = 3  # 1 essai initial + 2 retries
_RETRY_BACKOFF_S = 0.5  # backoff exponentiel : 0.5s, 1.0s entre retries
_OLLAMA_MODEL = "gemma3:4b"

_PLAN_PROMPT_TEMPLATE = (
    "Tu es un nutritionniste. Genere un plan repas JSON pour {duration_days} jours.\n"
    "Objectif : {objective}.\n"
    "Regime : {diet_type}.\n"
    "Allergies a eviter (aucun ingredient ne doit en contenir) : {allergies}.\n"
    "Cible calorique journaliere : {calories_target}.\n"
    "Pour chaque repas : name, macros (calories, protein_g, carbs_g, fat_g),\n"
    "ingredients (liste), est_budget_eur, prep_time_min. Mets fallback=false.\n"
    "Reponds uniquement par un JSON conforme au schema fourni."
)

_RECO_PROMPT_TEMPLATE = (
    "Tu es un coach nutritionnel. L'utilisateur a un desequilibre {imbalance} "
    "alors que son objectif est {health_goal}. Donne en une a deux phrases une "
    "recommandation actionnable, en francais, sans formattage markdown."
)

_RECO_DEFAULT_FALLBACK = (
    "Conseil indisponible pour le moment. Maintiens une alimentation equilibree."
)


# Helpers : canonicalisation et hash


def canonicalize_inputs(inputs: PlanInputs) -> dict[str, Any]:
    """Produit un dict deterministe : allergies triees, cles JSON ordonnees."""
    data = inputs.model_dump(mode="json")
    if data.get("allergies"):
        data["allergies"] = sorted(data["allergies"])
    return _sort_keys(data)


def _sort_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sort_keys(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_keys(v) for v in obj]
    return obj


def compute_inputs_hash(inputs: PlanInputs) -> str:
    """SHA256 hex du JSON canonicalise."""
    canonical = canonicalize_inputs(inputs)
    serialized = json.dumps(
        canonical, separators=(",", ":"), ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# Generation de plan repas


async def generate_plan(
    inputs: PlanInputs,
    db: Session,
    bypass_cache: bool = False,
    fallback_loader: Callable[[str, str], dict[str, Any] | None] | None = None,
) -> FallbackMealPlan:
    """Genere un plan repas via Ollama, avec cache, retry, semaphore, validation."""
    inputs_hash = compute_inputs_hash(inputs)

    if not bypass_cache:
        cached = _lookup_cached_plan(db, inputs.user_id, inputs_hash)
        if cached is not None:
            return cached

    try:
        plan = await _attempt_with_retry(
            prompt=_build_plan_prompt(inputs),
            json_schema=FallbackMealPlan.model_json_schema(),
            parse=lambda raw: _validate_plan(raw, inputs),
            log_id=inputs_hash,
        )
    except _OllamaCallFailed:
        return _build_fallback_plan(inputs, inputs_hash, db, fallback_loader)

    _persist_plan(db, plan, inputs_hash, inputs)
    return plan


# Generation de recommandation textuelle


async def generate_recommendation(
    ctx: RecommendationContext,
    db: Session,  # noqa: ARG001 (signature publique imposee par l'issue)
    fallback: Callable[[Imbalance, HealthGoal], str] | None = None,
) -> str:
    """Genere une recommandation textuelle courte via Ollama, avec retry et fallback."""
    prompt = _RECO_PROMPT_TEMPLATE.format(
        imbalance=ctx.imbalance.value, health_goal=ctx.health_goal.value
    )
    log_id = f"reco:{ctx.imbalance.value}:{ctx.health_goal.value}:{ctx.user_id}"

    try:
        return await _attempt_with_retry(
            prompt=prompt,
            json_schema=None,
            parse=_parse_text_response,
            log_id=log_id,
        )
    except _OllamaCallFailed:
        if fallback is not None:
            return fallback(ctx.imbalance, ctx.health_goal)
        return _RECO_DEFAULT_FALLBACK


# Internes : Ollama, validation, cache, fallback


class _OllamaCallFailed(Exception):
    """Levee apres _MAX_ATTEMPTS echecs : signal pour basculer en fallback."""


class _PlanValidationError(Exception):
    """Levee quand un plan viole une regle metier (allergie presente, JSON invalide)."""


async def _attempt_with_retry(
    prompt: str,
    json_schema: dict[str, Any] | None,
    parse: Callable[[str], T],
    log_id: str,
) -> T:
    """Boucle retry + semaphore commune a generate_plan / generate_recommendation."""
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        start = time.monotonic()
        try:
            async with _OLLAMA_SEMAPHORE:
                raw = await _call_ollama_generate(prompt, json_schema)
            result = parse(raw)
            _log_call(log_id, start, attempt, "success")
            return result
        except (httpx.HTTPError, ValidationError, _PlanValidationError) as exc:
            last_error = exc
            status = "retry" if attempt < _MAX_ATTEMPTS else "fallback"
            _log_call(log_id, start, attempt, status, error=str(exc))
            if attempt < _MAX_ATTEMPTS:
                # Backoff exponentiel : 0.5s, 1.0s. Laisse Ollama respirer.
                await asyncio.sleep(_RETRY_BACKOFF_S * (2 ** (attempt - 1)))
    _LOGGER.warning(
        "llm_client : echec apres %d tentatives (log_id=%s) : %s",
        _MAX_ATTEMPTS,
        log_id,
        last_error,
    )
    raise _OllamaCallFailed(str(last_error)) from last_error


def _parse_text_response(raw: str) -> str:
    text_reco = raw.strip()
    if not text_reco:
        raise _PlanValidationError("reponse vide.")
    return text_reco


async def _call_ollama_generate(prompt: str, json_schema: dict[str, Any] | None) -> str:
    """Appelle Ollama /api/generate et retourne la chaine 'response'.

    json_schema=None pour les sorties textes libres (pas de format=...).
    """
    payload: dict[str, Any] = {
        "model": _OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if json_schema is not None:
        payload["format"] = json_schema
    async with httpx.AsyncClient(timeout=_OLLAMA_TIMEOUT_S) as client:
        resp = await client.post(f"{settings.ollama_host}/api/generate", json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data.get("response", "")


def _validate_plan(raw_response: str, inputs: PlanInputs) -> FallbackMealPlan:
    """Parse le JSON, valide via Pydantic, applique les regles metier."""
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise _PlanValidationError(f"JSON invalide : {exc}") from exc
    # Le LLM oublie parfois 'fallback' qui est requis par le schema. On le
    # force a False sur le chemin success ; _build_fallback_plan le met a True.
    if isinstance(parsed, dict):
        parsed.setdefault("fallback", False)
    plan = FallbackMealPlan.model_validate(parsed)
    if inputs.allergies:
        spec = ConstraintSpec(allergies=inputs.allergies)
        for v in validate_constraints(plan, spec):
            if v.type is ViolationType.allergy:
                raise _PlanValidationError(v.message)
    return plan


def _build_plan_prompt(inputs: PlanInputs) -> str:
    return _PLAN_PROMPT_TEMPLATE.format(
        duration_days=inputs.duration_days,
        objective=inputs.objective,
        diet_type=inputs.diet_type or "aucun",
        allergies=", ".join(sorted(inputs.allergies)) if inputs.allergies else "aucune",
        calories_target=inputs.calories_target
        if inputs.calories_target
        else "non specifiee",
    )


def _lookup_cached_plan(
    db: Session, user_id: int, inputs_hash: str
) -> FallbackMealPlan | None:
    """Recupere le dernier plan en cache (< 7 jours) pour ce inputs_hash.

    Filtre redondant sur user_id : le hash inclut deja user_id, mais l'expliciter
    documente la requete et protege en defense-en-profondeur si la canonicalisation
    venait a changer.
    """
    row = db.execute(
        text(
            "SELECT plan FROM meal_plans "
            "WHERE user_id = :uid "
            "AND inputs_hash = :h "
            "AND generated_at > NOW() - INTERVAL '7 days' "
            "ORDER BY generated_at DESC "
            "LIMIT 1"
        ),
        {"uid": user_id, "h": inputs_hash},
    ).fetchone()
    if row is None:
        return None
    return FallbackMealPlan.model_validate(row.plan)


def _persist_plan(
    db: Session, plan: FallbackMealPlan, inputs_hash: str, inputs: PlanInputs
) -> None:
    """Insere une ligne meal_plans avec le plan complet et son inputs_hash.

    Flush sans commit : la decision de commit revient au caller (handler FastAPI),
    pour ne pas finaliser une transaction qui contient peut-etre d'autres ecritures.
    """
    db.execute(
        text(
            "INSERT INTO meal_plans (user_id, plan, objective, constraints, inputs_hash) "
            "VALUES (:uid, CAST(:plan AS JSONB), :obj, CAST(:cons AS JSONB), :h)"
        ),
        {
            "uid": inputs.user_id,
            "plan": plan.model_dump_json(),
            "obj": inputs.objective,
            "cons": json.dumps(canonicalize_inputs(inputs)),
            "h": inputs_hash,
        },
    )
    db.flush()


def _build_fallback_plan(
    inputs: PlanInputs,
    inputs_hash: str,
    db: Session,
    fallback_loader: Callable[[str, str], dict[str, Any] | None] | None,
) -> FallbackMealPlan:
    """Construit un plan en mode degrade : matrice statique ou squelette vide."""
    raw: dict[str, Any] | None = None
    if fallback_loader is not None:
        raw = fallback_loader(inputs.objective, inputs.diet_type or "")

    if raw is None:
        raw = {"days": []}
    raw["fallback"] = True

    plan = FallbackMealPlan.model_validate(raw)
    _persist_plan(db, plan, inputs_hash, inputs)
    return plan


# Logging structure


def _log_call(
    inputs_hash: str,
    start_monotonic: float,
    attempt: int,
    status: str,
    error: str | None = None,
) -> None:
    latency_ms = int((time.monotonic() - start_monotonic) * 1000)
    payload = {
        "inputs_hash": inputs_hash,
        "latency_ms": latency_ms,
        "attempt": attempt,
        "status": status,
    }
    if error:
        payload["error"] = error
    _LOGGER.info("llm_client.call %s", payload)
