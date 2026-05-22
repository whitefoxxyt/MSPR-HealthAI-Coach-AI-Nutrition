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
    ImbalanceTag,
    PlanInputs,
)
from app.services.decrim_retry_orchestrator import (
    ComplianceStatus,
    InfeasibleConstraintsError,
    generate_with_retry,
)
from app.services.fallback_chain import format_fallback_warning
from app.services.llm_provider import OllamaProvider

T = TypeVar("T")

_LOGGER = logging.getLogger(__name__)

# Limite la charge CPU d'Ollama : au plus 2 inferences simultanees, peu importe
# le nombre de coroutines qui appellent generate_plan.
_OLLAMA_SEMAPHORE = asyncio.Semaphore(2)

_OLLAMA_TIMEOUT_S = 30.0
_MAX_ATTEMPTS = 3  # 1 essai initial + 2 retries (flakiness Ollama)
_RETRY_BACKOFF_S = 0.5  # backoff exponentiel : 0.5s, 1.0s entre retries
_OLLAMA_MODEL = "gemma3:4b"

_RECO_PROMPT_TEMPLATE = (
    "Tu es un coach nutritionnel. L'utilisateur a les desequilibres suivants "
    "sur ce repas :\n{imbalances_block}\nSon objectif est {health_goal}. Donne "
    "une recommandation synthetique unique en deux a trois phrases, en francais, "
    "sans formattage markdown, qui couvre l'ensemble des desequilibres."
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
    primary_backend: str | None = None,
) -> tuple[FallbackMealPlan, ComplianceStatus, list[str]]:
    """Genere un plan repas en deleguant la boucle DeCRIM-light a l'orchestrator.

    Slice 7 PRD #45 : llm_client wrappe generate_with_retry pour gerer le cache,
    la flakiness Ollama (httpx erreurs / JSON invalide / Pydantic), la persistance
    et le fallback applicatif. La boucle de retry sur contraintes (allergie /
    regime / budget) vit dans decrim_retry_orchestrator.

    Retours :
      - (plan, status, warnings) tuple
      - InfeasibleConstraintsError remontee tel quel : le router traduit en 503

    Comportement flakiness : 3 tentatives orchestrateur. Chaque tentative qui
    leve httpx/JSON/Pydantic compte pour 1. Apres _MAX_ATTEMPTS echecs ou si la
    1ere tentative reussit avec status non-full, on s'arrete.
    """
    inputs_hash = compute_inputs_hash(inputs)
    primary_backend = primary_backend or settings.default_llm

    if not bypass_cache:
        cached = _lookup_cached_plan(db, inputs.user_id, inputs_hash, primary_backend)
        if cached is not None:
            return cached

    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        start = time.monotonic()
        try:
            async with _OLLAMA_SEMAPHORE:
                plan, status, used_secondary = await generate_with_retry(
                    inputs, primary_backend=primary_backend
                )
        except InfeasibleConstraintsError:
            # On trace l'attempt avant de remonter pour que les ops puissent
            # correler la latence et l'inputs_hash avec le 503 cote router.
            _log_call(inputs_hash, start, attempt, "infeasible")
            raise
        except (httpx.HTTPError, json.JSONDecodeError, ValidationError) as exc:
            last_error = exc
            log_status = "retry" if attempt < _MAX_ATTEMPTS else "fallback"
            _log_call(inputs_hash, start, attempt, log_status, error=str(exc))
            if attempt < _MAX_ATTEMPTS:
                # Backoff exponentiel : 0.5s, 1.0s. Laisse Ollama respirer.
                await asyncio.sleep(_RETRY_BACKOFF_S * (2 ** (attempt - 1)))
            continue

        warnings = _build_compliance_warnings(plan, status, inputs)
        # Le backend reellement utilise est used_secondary si la chain a fallback,
        # sinon le primaire. C'est lui qu'on persiste et qu'on filtrera au cache.
        backend_used = used_secondary or primary_backend
        if used_secondary is not None:
            warnings.insert(0, format_fallback_warning(primary_backend, used_secondary))
        _persist_plan(
            db, plan, inputs_hash, inputs, status.value, warnings, backend_used
        )
        _log_call(inputs_hash, start, attempt, "success")
        return plan, status, warnings

    _LOGGER.warning(
        "llm_client : echec apres %d tentatives (log_id=%s) : %s",
        _MAX_ATTEMPTS,
        inputs_hash,
        last_error,
    )
    return _build_fallback_plan(
        inputs, inputs_hash, db, fallback_loader, primary_backend
    )


# Generation de recommandation textuelle


async def generate_recommendation(
    ctx_list: list[ImbalanceTag],
    health_goal: HealthGoal,
    db: Session,  # noqa: ARG001 (reservee pour cache cote orchestrator)
    fallback: Callable[[list[ImbalanceTag], HealthGoal], str] | None = None,
) -> str:
    """Genere une recommandation synthetique unique pour l'ensemble des imbalances.

    Issue #51 : un seul appel Ollama, prompt couvrant toutes les imbalances et
    l'objectif sante. Le cache est gere par meal_analysis_orchestrator (cle :
    hash de top_label + health_goal + imbalances triees).
    """
    if not ctx_list:
        return ""

    sorted_tags = sorted(ctx_list, key=_tag_sort_key)
    prompt = _build_recommendation_prompt(sorted_tags, health_goal)
    log_id = (
        "reco:"
        + ",".join(f"{t.nutrient.value}/{t.status.value}" for t in sorted_tags)
        + f":{health_goal.value}"
    )

    try:
        return await _attempt_with_retry(
            prompt=prompt,
            json_schema=None,
            parse=_parse_text_response,
            log_id=log_id,
        )
    except _OllamaCallFailed:
        if fallback is not None:
            return fallback(sorted_tags, health_goal)
        return _RECO_DEFAULT_FALLBACK


def _tag_sort_key(tag: ImbalanceTag) -> tuple[str, str]:
    return (tag.nutrient.value, tag.status.value)


def _build_recommendation_prompt(
    tags: list[ImbalanceTag], health_goal: HealthGoal
) -> str:
    # Ordre stable -> prompt deterministe -> reponse cache-friendly.
    lines = [
        f"- {tag.nutrient.value} ({tag.status.value}, ecart "
        f"{tag.delta_pct * 100:+.0f}%, cible {tag.target_value:.1f} {tag.unit}, "
        f"apport {tag.actual_value:.1f} {tag.unit})"
        for tag in tags
    ]
    return _RECO_PROMPT_TEMPLATE.format(
        imbalances_block="\n".join(lines),
        health_goal=health_goal.value,
    )


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
    """Boucle retry + semaphore pour generate_recommendation.

    generate_plan delegue desormais a decrim_retry_orchestrator et n'utilise
    plus cette helper. Conservee pour les recommandations textuelles libres.
    """
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
    """Genere une recommandation textuelle via OllamaProvider.

    Config dediee : timeout court (30s, recos rapides) et pas de num_predict
    (les recos sont courtes, on laisse Ollama gerer sa terminaison). Cette
    path reste Ollama-only ; le selecteur multi-provider concerne le plan repas.
    """
    provider = OllamaProvider(
        base_url=settings.ollama_host,
        timeout=_OLLAMA_TIMEOUT_S,
        num_predict=None,
    )
    return await provider.generate(prompt, json_schema)


def _build_compliance_warnings(
    plan: FallbackMealPlan,
    status: ComplianceStatus,
    inputs: PlanInputs,
) -> list[str]:
    """Genere les warnings explicites associes au compliance_status.

    full           : aucune contrainte relachee.
    partial_budget : enumere les jours qui depassent le budget journalier.
    static_fallback: explique que le plan statique sert de repli.
    """
    if status is ComplianceStatus.full:
        return []
    if status is ComplianceStatus.static_fallback:
        return ["Contraintes infaisables par le LLM, plan statique de repli."]
    if status is ComplianceStatus.partial_budget:
        budget = float(inputs.budget_per_day) if inputs.budget_per_day else None
        if budget is None:
            return ["Plan partiellement conforme (budget)."]
        warnings: list[str] = []
        for day in plan.days:
            day_total = sum(meal.est_budget_eur for meal in day.meals)
            if day_total > budget:
                overflow = day_total - budget
                warnings.append(
                    f"Budget journalier depasse de {overflow:.2f} EUR le jour {day.day}."
                )
        return warnings or ["Plan partiellement conforme (budget)."]
    return []


def _lookup_cached_plan(
    db: Session, user_id: str, inputs_hash: str, backend: str
) -> tuple[FallbackMealPlan, ComplianceStatus, list[str]] | None:
    """Recupere le dernier plan en cache (< 7 jours) pour ce (user, hash, backend).

    Filtre par backend (PRD #71 slice 3) : un switch user Ollama -> Mistral
    doit produire un nouveau plan plutot que servir le cache cross-backend.

    Renvoie aussi compliance_status et compliance_warnings tels que persistes.
    Filtre redondant sur user_id : le hash inclut deja user_id, mais l'expliciter
    documente la requete et protege en defense-en-profondeur si la canonicalisation
    venait a changer.
    """
    row = db.execute(
        text(
            "SELECT plan, compliance_status, compliance_warnings FROM meal_plans "
            "WHERE user_id = :uid "
            "AND inputs_hash = :h "
            "AND llm_backend_used = :b "
            "AND generated_at > NOW() - INTERVAL '7 days' "
            "ORDER BY generated_at DESC "
            "LIMIT 1"
        ),
        {"uid": user_id, "h": inputs_hash, "b": backend},
    ).fetchone()
    if row is None:
        return None
    plan = FallbackMealPlan.model_validate(row.plan)
    status = ComplianceStatus(row.compliance_status or "full")
    warnings = list(row.compliance_warnings or [])
    return plan, status, warnings


def _persist_plan(
    db: Session,
    plan: FallbackMealPlan,
    inputs_hash: str,
    inputs: PlanInputs,
    compliance_status: str,
    compliance_warnings: list[str],
    llm_backend_used: str,
) -> None:
    """Insere une ligne meal_plans avec le plan complet, son inputs_hash et le compliance.

    Flush sans commit : la decision de commit revient au caller (handler FastAPI),
    pour ne pas finaliser une transaction qui contient peut-etre d'autres ecritures.
    """
    db.execute(
        text(
            "INSERT INTO meal_plans (user_id, plan, objective, constraints, "
            "inputs_hash, compliance_status, compliance_warnings, llm_backend_used) "
            "VALUES (:uid, CAST(:plan AS JSONB), :obj, CAST(:cons AS JSONB), :h, "
            ":status, :warnings, :backend)"
        ),
        {
            "uid": inputs.user_id,
            "plan": plan.model_dump_json(),
            "obj": inputs.objective,
            "cons": json.dumps(canonicalize_inputs(inputs)),
            "h": inputs_hash,
            "status": compliance_status,
            "warnings": list(compliance_warnings or []),
            "backend": llm_backend_used,
        },
    )
    db.flush()


def _build_fallback_plan(
    inputs: PlanInputs,
    inputs_hash: str,
    db: Session,
    fallback_loader: Callable[[str, str], dict[str, Any] | None] | None,
    primary_backend: str,
) -> tuple[FallbackMealPlan, ComplianceStatus, list[str]]:
    """Construit un plan en mode degrade : matrice statique ou squelette vide.

    Cas LLM injoignable apres tous les retries flakiness (Slice 2 : la chain
    Mistral <-> Ollama a aussi echoue sur chaque tentative). Le plan retourne
    porte status=static_fallback et un warning explicitant la cause.

    On persiste avec primary_backend : le plan statique n'a pas ete genere par
    un LLM, mais on attribue la ligne au backend "demande" pour la coherence du
    filtre cache (un switch user re-tentera l'autre backend plutot que servir
    ce static fallback en cache).
    """
    raw: dict[str, Any] | None = None
    if fallback_loader is not None:
        raw = fallback_loader(inputs.objective, inputs.diet_type or "")

    if raw is None:
        raw = {"days": []}
    raw["fallback"] = True

    plan = FallbackMealPlan.model_validate(raw)
    warnings = ["LLM injoignable, plan statique de repli."]
    _persist_plan(
        db,
        plan,
        inputs_hash,
        inputs,
        ComplianceStatus.static_fallback.value,
        warnings,
        primary_backend,
    )
    return plan, ComplianceStatus.static_fallback, warnings


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
