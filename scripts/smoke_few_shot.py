"""Smoke test : valide que le prompt few-shot condense passe sur CPU.

Appelle directement _ollama_generate de l'orchestrator (skip BDD/persistance).
Mesure 3 plans, affiche la longueur du prompt, la latence et un extrait du JSON.

Usage (dans container ai-nutrition, reseau ai_internal) :
    OLLAMA_HOST=http://mspr-ollama:11434 python scripts/smoke_few_shot.py
"""

from __future__ import annotations

import asyncio
import time

from app.models.schemas import FallbackMealPlan, PlanInputs
from app.services.decrim_retry_orchestrator import (
    _build_plan_prompt,
    _ollama_generate,
)


async def main() -> None:
    inputs_list = [
        PlanInputs(user_id=1, objective="balance", duration_days=1),
        PlanInputs(
            user_id=2,
            objective="weight_loss",
            duration_days=1,
            diet_type="omnivore",
            allergies=["arachides"],
        ),
        PlanInputs(
            user_id=3,
            objective="muscle_gain",
            duration_days=1,
            diet_type="vegetarian",
            calories_target=2500,
        ),
    ]
    schema = FallbackMealPlan.model_json_schema()

    for i, inputs in enumerate(inputs_list, 1):
        prompt = _build_plan_prompt(inputs)
        print(f"--- Run {i}/3 ---")
        print(f"  Prompt length : {len(prompt)} chars")
        start = time.monotonic()
        try:
            response = await _ollama_generate(prompt, schema)
            elapsed = time.monotonic() - start
            print(f"  OK : {elapsed:.1f}s, response {len(response)} chars")
            print(f"  Preview : {response[:200]}...")
        except Exception as exc:
            elapsed = time.monotonic() - start
            print(f"  FAIL after {elapsed:.1f}s : {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
