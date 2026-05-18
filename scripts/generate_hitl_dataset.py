"""Genere 20 plans repas pour annotation HITL (humaine ou LLM-as-Judge).

Utilise le pipeline complet `generate_plan` (decrim_retry_orchestrator + cache
bypass) avec 20 PlanInputs aleatoires reproductibles (seed=42), exactement les
memes que ceux utilises par scripts/eval/llm_runner._run_pipeline_eval. Cela
garantit que l'eval HITL et l'eval automatique mesurent le meme echantillon.

Sorties :
  data/hitl/plans.jsonl              un plan complet par ligne (inputs + plan + compliance)
  data/hitl/plans.md                 rendu Markdown lisible pour l'annotation manuelle
  data/hitl/ratings_template.csv     template a remplir (plan_id, nutrition, originalite, coherence)

Utilisation :
  docker exec mspr-ai-nutrition python scripts/generate_hitl_dataset.py

Pre-requis : Ollama et PostgreSQL accessibles (voir docs/GPU_EVAL_PLAYBOOK.md).
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import random
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.db.session import SessionLocal
from app.models.schemas import DietType, FallbackMealPlan, HealthGoal, PlanInputs
from app.services.decrim_retry_orchestrator import InfeasibleConstraintsError
from app.services.fallback_loader import load_fallback_plan
from app.services.llm_client import generate_plan

logger = logging.getLogger(__name__)

_N_PLANS = 20
_SEED = 42

# Memes pools que scripts/eval/llm_runner._random_inputs pour la coherence eval.
_GOALS = [g.value for g in HealthGoal]
_DIETS = [d.value for d in DietType]
_ALLERGEN_POOL = ["arachide", "lait", "gluten", "oeuf", "soja", "fruits a coque"]


def _random_inputs(rng: random.Random, idx: int) -> PlanInputs:
    """Reproduit la logique de llm_runner._random_inputs(with_constraints=True)."""
    objective = rng.choice(_GOALS)
    diet = rng.choice(_DIETS)
    allergies = rng.sample(_ALLERGEN_POOL, k=rng.randint(1, 2))
    budget = Decimal(str(round(rng.uniform(8.0, 20.0), 2)))
    return PlanInputs(
        user_id=1_000_000 + idx,
        objective=objective,
        duration_days=1,
        diet_type=diet,
        allergies=allergies,
        budget_per_day=budget,
    )


async def _generate_one(idx: int, inputs: PlanInputs) -> dict[str, Any]:
    """Genere un plan via le pipeline complet, retourne le record HITL."""
    db = SessionLocal()
    try:
        try:
            plan, status, warnings = await generate_plan(
                inputs,
                db,
                bypass_cache=True,
                fallback_loader=load_fallback_plan,
            )
            return {
                "plan_id": idx + 1,
                "inputs": inputs.model_dump(mode="json"),
                "compliance_status": status.value,
                "compliance_warnings": list(warnings),
                "plan": plan.model_dump(mode="json"),
            }
        except InfeasibleConstraintsError as exc:
            return {
                "plan_id": idx + 1,
                "inputs": inputs.model_dump(mode="json"),
                "compliance_status": "abandoned_503",
                "compliance_warnings": [str(exc)],
                "plan": None,
            }
    finally:
        db.rollback()
        db.close()


def _render_markdown(records: list[dict[str, Any]]) -> str:
    """Rendu lisible des plans pour annotation manuelle (Colab, editeur, etc.)."""
    lines: list[str] = []
    lines.append("# Plans HITL a annoter")
    lines.append("")
    lines.append(
        "Notez chaque plan sur trois dimensions (echelle 1-5), en aveugle :"
    )
    lines.append("")
    lines.append("- nutrition : le plan est-il nutritionnellement coherent ?")
    lines.append("- originalite : les repas sont-ils varies, pas repetitifs ?")
    lines.append(
        "- coherence : les ingredients et le nom du repas correspondent-ils ?"
    )
    lines.append("")
    lines.append(
        "Ne lisez pas le bloc Contraintes avant d'avoir note (biais de confirmation)."
    )
    lines.append("")
    for record in records:
        lines.append(f"## Plan {record['plan_id']}")
        lines.append("")
        plan = record["plan"]
        if plan is None:
            lines.append(
                f"**Plan non genere** (compliance_status = {record['compliance_status']})."
            )
            lines.append("")
            continue
        for day in plan["days"]:
            lines.append(f"### Jour {day['day']}")
            for meal in day["meals"]:
                macros = meal["macros"]
                lines.append(f"- **{meal['name']}**")
                lines.append(
                    f"  - macros : {macros['calories']} kcal, "
                    f"{macros['protein_g']} g P / {macros['carbs_g']} g G / "
                    f"{macros['fat_g']} g L"
                )
                ingredients = ", ".join(meal["ingredients"])
                lines.append(f"  - ingredients : {ingredients}")
                lines.append(
                    f"  - budget {meal['est_budget_eur']} EUR, "
                    f"prep {meal['prep_time_min']} min"
                )
            lines.append("")
        lines.append("<details><summary>Contraintes (a ne pas lire avant la note)</summary>")
        lines.append("")
        inputs = record["inputs"]
        lines.append(f"- objectif : {inputs['objective']}")
        lines.append(f"- regime : {inputs.get('diet_type') or 'aucun'}")
        allergies = inputs.get("allergies") or []
        lines.append(
            f"- allergies : {', '.join(allergies) if allergies else 'aucune'}"
        )
        lines.append(f"- budget journalier : {inputs.get('budget_per_day')} EUR")
        lines.append(f"- compliance_status : {record['compliance_status']}")
        warnings = record["compliance_warnings"]
        if warnings:
            lines.append(f"- warnings : {' ; '.join(warnings)}")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output_dir = Path("data/hitl")
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(_SEED + 1)  # Aligne avec llm_runner _run_pipeline_eval (seed+1)
    inputs_list = [_random_inputs(rng, i) for i in range(_N_PLANS)]

    records: list[dict[str, Any]] = []
    for i, inputs in enumerate(inputs_list):
        logger.info("hitl : generation %d/%d (objective=%s, diet=%s)",
                    i + 1, _N_PLANS, inputs.objective, inputs.diet_type)
        record = await _generate_one(i, inputs)
        records.append(record)

    # JSONL : 1 ligne par plan, format canonique pour traitement programmatique.
    jsonl_path = output_dir / "plans.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("hitl : ecrit %s (%d plans)", jsonl_path, len(records))

    # Markdown : version lisible pour annotation manuelle.
    md_path = output_dir / "plans.md"
    md_path.write_text(_render_markdown(records), encoding="utf-8")
    logger.info("hitl : ecrit %s", md_path)

    # CSV template : a copier vers docs/llm_hitl_ratings.csv apres annotation.
    csv_path = output_dir / "ratings_template.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["plan_id", "nutrition", "originalite", "coherence"])
        for record in records:
            writer.writerow([record["plan_id"], "", "", ""])
    logger.info("hitl : ecrit %s", csv_path)

    logger.info(
        "Prochaine etape : annoter %s, copier vers docs/llm_hitl_ratings.csv, "
        "puis relancer python scripts/eval_metrics.py llm pour integrer les ratings.",
        csv_path,
    )


if __name__ == "__main__":
    asyncio.run(main())
