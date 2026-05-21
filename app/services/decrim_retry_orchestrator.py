from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from enum import Enum
from typing import Any

from app.config import settings
from app.data.plan_few_shot_examples import FEW_SHOT_EXAMPLES, FewShotExample
from app.models.schemas import FallbackMealPlan, Meal, MealDay, PlanInputs
from app.services.constraint_validator import (
    ConstraintSpec,
    ConstraintViolation,
    ViolationType,
    validate as validate_constraints,
)
from app.services.fallback_chain import FallbackChain, build_default_chain
from app.services.fallback_loader import load_fallback_plan

LLMGen = Callable[[str, dict[str, Any]], Awaitable[str]]


def _format_few_shot_example(idx: int, example: FewShotExample) -> str:
    """Formate un exemple en bloc texte : entete + JSON (1er jour) + annotation.

    Slice volontaire a days[:1] : Gemma3:4b sur CPU timeout au-dela de ~2000
    tokens de prefill. Inclure le 1er jour suffit a transmettre la structure
    attendue (schema + style) sans exploser le prompt. Les FEW_SHOT_EXAMPLES
    restent complets (7j/5j/1j) pour les tests et un futur run GPU.
    """
    verdict = "valide" if example.is_valid else "rejete"
    header = f"Exemple {idx} ({example.label}, {verdict}) :"
    sliced = example.plan.model_copy(update={"days": example.plan.days[:1]})
    plan_json = sliced.model_dump_json()
    if example.is_valid or not example.rejection_reason:
        return f"{header}\n{plan_json}"
    return f"{header}\n{plan_json}\nMotif du rejet : {example.rejection_reason}"


_FEW_SHOT_BLOCK = "\n\n".join(
    _format_few_shot_example(i + 1, ex) for i, ex in enumerate(FEW_SHOT_EXAMPLES)
)
# Echappe les accolades du JSON pour str.format() applique plus bas.
_FEW_SHOT_BLOCK_ESCAPED = _FEW_SHOT_BLOCK.replace("{", "{{").replace("}", "}}")

_CORE_INSTRUCTION = (
    "genere un plan repas JSON pour {duration_days} jours.\n"
    "Objectif : {objective}.\n"
    "Regime : {diet_type}.\n"
    "Allergies a eviter (aucun ingredient ne doit en contenir) : {allergies}.\n"
    "Budget journalier maximal : {budget}.\n"
    "Cible calorique journaliere : {calories_target}.\n"
    "Pour chaque repas : name, macros (calories, protein_g, carbs_g, fat_g),\n"
    "ingredients (liste), est_budget_eur, prep_time_min. Mets fallback=false."
)

_PLAN_PROMPT_TEMPLATE = (
    "Tu es un nutritionniste. Voici 3 exemples de plans valides ou rejetes :\n"
    f"{_FEW_SHOT_BLOCK_ESCAPED}\n\n"
    f"Maintenant {_CORE_INSTRUCTION}"
)

_PLAN_PROMPT_TEMPLATE_NO_FEW_SHOT = (
    f"Tu es un nutritionniste. {_CORE_INSTRUCTION[0].upper()}{_CORE_INSTRUCTION[1:]}"
)

_MEAL_REGEN_PROMPT_TEMPLATE = (
    "Ce repas a ete rejete car il contient {ingredient!r}. "
    "Regenere-le sans {ingredient!r}. "
    "Respecte le regime {diet_type} et evite ces allergenes : {allergies}. "
    "Reponds par un seul JSON Meal : name, macros (calories, protein_g, "
    "carbs_g, fat_g), ingredients (liste), est_budget_eur, prep_time_min."
)

_DAY_REGEN_PROMPT_TEMPLATE = (
    "Ce jour depasse le budget de {overflow:.2f} EUR. "
    "Regenere ce jour sous {budget_max:.2f} EUR. "
    "Garde {meals_count} repas et respecte le regime {diet_type}. "
    "Reponds par un seul JSON MealDay : day (entier), meals (liste de Meal "
    "avec name, macros, ingredients, est_budget_eur, prep_time_min)."
)


class ComplianceStatus(str, Enum):
    full = "full"
    partial_budget = "partial_budget"
    static_fallback = "static_fallback"


class InfeasibleConstraintsError(Exception):
    """Allergies / regime infaisables : meme le plan statique de fallback les viole.

    Le caller (router FastAPI) traduit en HTTP 503.
    """


async def generate_with_retry(
    inputs: PlanInputs,
    max_retries: int = 3,
    chain: FallbackChain | None = None,
    primary_backend: str | None = None,
) -> tuple[FallbackMealPlan, ComplianceStatus, str | None]:
    """Genere un plan repas avec retries DeCRIM-light hybride.

    Strategie :
    - allergie / regime : retry partiel sur le repas violant uniquement
    - budget : retry complet du jour qui depasse

    Retour : (plan, status, used_secondary). `used_secondary` est le nom du
    backend secondaire effectivement utilise quand le primaire a echoue
    (issue #73 / PRD #71 slice 2), None si le primaire a repondu sur tous
    les appels. Permet au caller d'ajouter un compliance_warning explicite.
    """
    primary = primary_backend or settings.llm_backend
    active_chain = chain or build_default_chain()
    used_backends: list[str] = []

    async def llm_gen(prompt: str, schema: dict[str, Any]) -> str:
        content, used = await active_chain.generate(prompt, schema, primary)
        used_backends.append(used)
        return content

    spec = _build_constraint_spec(inputs)
    plan = await _call_for_plan(_build_plan_prompt(inputs), llm_gen)

    # Compteur de retries partiels par (jour, repas) : permet de detecter
    # un cycle ou la meme position viole encore apres 2 regenerations ciblees.
    meal_retry_counts: dict[tuple[int, int], int] = {}

    for _ in range(max_retries):
        violations = validate_constraints(plan, spec)
        if not violations:
            return (
                plan,
                ComplianceStatus.full,
                _pick_secondary_used(used_backends, primary),
            )

        first = violations[0]
        if first.type in (ViolationType.allergy, ViolationType.diet):
            meal_idx = first.meal_index if first.meal_index is not None else 0
            key = (first.day, meal_idx)
            if meal_retry_counts.get(key, 0) >= 2:
                # Garde-fou anti-cycle : la meme position viole encore apres 2
                # regenerations ciblees. On bascule en retry complet du plan.
                plan = await _call_for_plan(_build_plan_prompt(inputs), llm_gen)
                meal_retry_counts.clear()
                continue
            meal_retry_counts[key] = meal_retry_counts.get(key, 0) + 1
            new_meal = await _call_for_meal(
                _build_meal_regen_prompt(first, inputs), llm_gen
            )
            plan = _replace_meal(plan, first.day, meal_idx, new_meal)
            continue

        if first.type is ViolationType.budget:
            new_day = await _call_for_day(
                _build_day_regen_prompt(plan, first, spec), llm_gen
            )
            plan = _replace_day(plan, first.day, new_day)
            continue

    plan, status = _resolve_after_retries(plan, inputs, spec)
    return plan, status, _pick_secondary_used(used_backends, primary)


def _pick_secondary_used(used_backends: list[str], primary: str) -> str | None:
    """Premier backend non-primaire effectivement utilise, sinon None."""
    return next((b for b in used_backends if b != primary), None)


def _resolve_after_retries(
    plan: FallbackMealPlan, inputs: PlanInputs, spec: ConstraintSpec
) -> tuple[FallbackMealPlan, ComplianceStatus]:
    """Apres epuisement des retries : decide entre full / partial_budget / fallback statique."""
    remaining = validate_constraints(plan, spec)
    if not remaining:
        return plan, ComplianceStatus.full

    has_blocking = any(
        v.type in (ViolationType.allergy, ViolationType.diet) for v in remaining
    )
    if not has_blocking:
        # Seul le budget est viole : on conserve le plan LLM, charge au caller
        # de logger des warnings (cf. issue #52 : "warnings explicites").
        return plan, ComplianceStatus.partial_budget

    # Fallback hierarchique : plan statique correspondant a (objectif, regime).
    raw = load_fallback_plan(inputs.objective, inputs.diet_type or "")
    if raw is None:
        raise InfeasibleConstraintsError(
            f"Pas de plan statique pour {inputs.objective} / {inputs.diet_type}, "
            "impossible de satisfaire allergies/regime."
        )
    fallback = FallbackMealPlan.model_validate(raw)
    fb_violations = validate_constraints(fallback, spec)
    if any(
        v.type in (ViolationType.allergy, ViolationType.diet) for v in fb_violations
    ):
        raise InfeasibleConstraintsError(
            "Le plan statique de fallback viole les allergies / regime."
        )
    if any(v.type is ViolationType.budget for v in fb_violations):
        return fallback, ComplianceStatus.partial_budget
    return fallback, ComplianceStatus.static_fallback


def _build_constraint_spec(inputs: PlanInputs) -> ConstraintSpec:
    budget = float(inputs.budget_per_day) if inputs.budget_per_day is not None else None
    return ConstraintSpec(
        allergies=list(inputs.allergies),
        max_daily_budget_eur=budget,
        diet_type=inputs.diet_type,
    )


def _build_plan_prompt(inputs: PlanInputs) -> str:
    template = (
        _PLAN_PROMPT_TEMPLATE
        if settings.few_shot_enabled
        else _PLAN_PROMPT_TEMPLATE_NO_FEW_SHOT
    )
    return template.format(
        duration_days=inputs.duration_days,
        objective=inputs.objective,
        diet_type=inputs.diet_type or "aucun",
        allergies=", ".join(sorted(inputs.allergies)) if inputs.allergies else "aucune",
        budget=f"{inputs.budget_per_day} EUR"
        if inputs.budget_per_day
        else "non specifie",
        calories_target=inputs.calories_target or "non specifiee",
    )


def _build_meal_regen_prompt(violation: ConstraintViolation, inputs: PlanInputs) -> str:
    return _MEAL_REGEN_PROMPT_TEMPLATE.format(
        ingredient=str(violation.ingredient_or_amount),
        diet_type=inputs.diet_type or "aucun",
        allergies=", ".join(sorted(inputs.allergies)) if inputs.allergies else "aucune",
    )


def _build_day_regen_prompt(
    plan: FallbackMealPlan,
    violation: ConstraintViolation,
    spec: ConstraintSpec,
) -> str:
    day = next(d for d in plan.days if d.day == violation.day)
    budget_max = spec.max_daily_budget_eur or 0.0
    day_total = float(violation.ingredient_or_amount)
    return _DAY_REGEN_PROMPT_TEMPLATE.format(
        overflow=day_total - budget_max,
        budget_max=budget_max,
        meals_count=len(day.meals),
        diet_type=spec.diet_type or "aucun",
    )


async def _call_for_plan(prompt: str, llm_gen: LLMGen) -> FallbackMealPlan:
    raw = await llm_gen(prompt, FallbackMealPlan.model_json_schema())
    parsed = json.loads(raw)
    if isinstance(parsed, dict):
        parsed.setdefault("fallback", False)
    return FallbackMealPlan.model_validate(parsed)


async def _call_for_meal(prompt: str, llm_gen: LLMGen) -> Meal:
    raw = await llm_gen(prompt, Meal.model_json_schema())
    return Meal.model_validate_json(raw)


async def _call_for_day(prompt: str, llm_gen: LLMGen) -> MealDay:
    raw = await llm_gen(prompt, MealDay.model_json_schema())
    return MealDay.model_validate_json(raw)


def _replace_meal(
    plan: FallbackMealPlan, day_num: int, meal_idx: int, new_meal: Meal
) -> FallbackMealPlan:
    new_days: list[MealDay] = []
    for d in plan.days:
        if d.day != day_num:
            new_days.append(d)
            continue
        new_meals = list(d.meals)
        new_meals[meal_idx] = new_meal
        new_days.append(MealDay(day=d.day, meals=new_meals))
    return FallbackMealPlan(fallback=plan.fallback, days=new_days)


def _replace_day(
    plan: FallbackMealPlan, day_num: int, new_day: MealDay
) -> FallbackMealPlan:
    # Force le numero de jour : le LLM peut renvoyer un day=1 sur un sous-prompt.
    new_days = [
        MealDay(day=day_num, meals=new_day.meals) if d.day == day_num else d
        for d in plan.days
    ]
    return FallbackMealPlan(fallback=plan.fallback, days=new_days)
