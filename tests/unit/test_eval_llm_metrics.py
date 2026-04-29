from __future__ import annotations

from pathlib import Path

from app.models.schemas import FallbackMealPlan
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


def _plan(
    *,
    ingredients: list[str] = ("riz", "poulet"),
    cost_eur: float = 5.0,
) -> FallbackMealPlan:
    return FallbackMealPlan.model_validate(
        {
            "fallback": False,
            "days": [
                {
                    "day": 1,
                    "meals": [
                        {
                            "name": "lunch",
                            "macros": {
                                "calories": 500,
                                "protein_g": 30,
                                "carbs_g": 60,
                                "fat_g": 15,
                            },
                            "ingredients": list(ingredients),
                            "est_budget_eur": cost_eur,
                            "prep_time_min": 20,
                        }
                    ],
                }
            ],
        }
    )


def test_latency_percentiles_simple_distribution() -> None:
    # 100 valeurs : p50=50ms, p95=95ms, max=100ms
    latencies = [float(i) for i in range(1, 101)]

    result = latency_percentiles(latencies)

    # Avec methode "nearest-rank", p50 et p95 tombent precisement sur des valeurs.
    assert result["p50_ms"] == 50.0
    assert result["p95_ms"] == 95.0
    assert result["max_ms"] == 100.0


def test_latency_percentiles_single_value() -> None:
    result = latency_percentiles([42.0])

    assert result == {"p50_ms": 42.0, "p95_ms": 42.0, "max_ms": 42.0}


def test_latency_percentiles_empty_returns_zeros() -> None:
    # Une eval LLM peut tomber a zero appel reussi : ne pas lever, retourner 0.
    result = latency_percentiles([])

    assert result == {"p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}


def _outcome(
    *,
    json_valid: bool = True,
    used_fallback: bool = False,
    latency_ms: float = 100.0,
) -> GenerationOutcome:
    return GenerationOutcome(
        json_valid_first_try=json_valid,
        used_fallback=used_fallback,
        latency_ms=latency_ms,
    )


def test_json_validity_rate_counts_first_try_successes() -> None:
    outcomes = [
        _outcome(json_valid=True),
        _outcome(json_valid=False),
        _outcome(json_valid=True),
        _outcome(json_valid=True),
    ]

    assert json_validity_rate(outcomes) == 0.75


def test_json_validity_rate_empty_is_zero() -> None:
    assert json_validity_rate([]) == 0.0


def test_fallback_rate_counts_fallback_invocations() -> None:
    outcomes = [
        _outcome(used_fallback=False),
        _outcome(used_fallback=True),
        _outcome(used_fallback=True),
        _outcome(used_fallback=False),
    ]

    assert fallback_rate(outcomes) == 0.5


def test_fallback_rate_empty_is_zero() -> None:
    assert fallback_rate([]) == 0.0


def test_constraint_satisfaction_rate_all_three_must_pass() -> None:
    # Plan respecte : pas d'allergene, dans le budget, regime conforme
    ok = ConstraintCheck(
        allergies_absent=True,
        budget_respected=True,
        diet_respected=True,
    )
    # Chaque plan ne respectant pas l'une des 3 contraintes ne compte pas
    one_fail = ConstraintCheck(
        allergies_absent=True,
        budget_respected=False,
        diet_respected=True,
    )

    assert constraint_satisfaction_rate([ok, ok, one_fail, ok]) == 0.75


def test_constraint_satisfaction_rate_empty_is_zero() -> None:
    assert constraint_satisfaction_rate([]) == 0.0


def test_check_plan_constraints_all_satisfied() -> None:
    plan = _plan(ingredients=["riz", "poulet"], cost_eur=4.0)
    spec = ConstraintSpec(
        allergies=["arachide"],
        max_daily_budget_eur=10.0,
        diet_type="omnivore",
    )

    result = check_plan_constraints(plan, spec)

    assert result.allergies_absent is True
    assert result.budget_respected is True
    assert result.diet_respected is True


def test_check_plan_constraints_detects_allergen_word_boundary() -> None:
    # 'lait' figure dans la liste mais 'laitue' ne doit PAS matcher (faux positif).
    plan_safe = _plan(ingredients=["laitue", "tomate"])
    plan_unsafe = _plan(ingredients=["lait ecreme", "cafe"])
    spec = ConstraintSpec(
        allergies=["lait"], max_daily_budget_eur=10.0, diet_type="omnivore"
    )

    assert check_plan_constraints(plan_safe, spec).allergies_absent is True
    assert check_plan_constraints(plan_unsafe, spec).allergies_absent is False


def test_check_plan_constraints_budget_exceeded() -> None:
    # Le cout journalier somme tous les repas : ici 1 jour * 1 repas * 12 EUR.
    plan = _plan(cost_eur=12.0)
    spec = ConstraintSpec(allergies=[], max_daily_budget_eur=10.0, diet_type="omnivore")

    assert check_plan_constraints(plan, spec).budget_respected is False


def test_check_plan_constraints_vegan_rejects_animal_ingredients() -> None:
    plan_meat = _plan(ingredients=["poulet", "riz"])
    plan_vegan = _plan(ingredients=["tofu", "riz"])
    spec = ConstraintSpec(allergies=[], max_daily_budget_eur=100.0, diet_type="vegan")

    assert check_plan_constraints(plan_meat, spec).diet_respected is False
    assert check_plan_constraints(plan_vegan, spec).diet_respected is True


def test_load_hitl_ratings_aggregates_means(tmp_path: Path) -> None:
    csv = tmp_path / "hitl.csv"
    csv.write_text(
        "plan_id,nutrition,originalite,coherence\n1,5,3,4\n2,3,4,5\n3,4,4,3\n"
    )

    summary = load_hitl_ratings(csv)

    assert summary.n_ratings == 3
    assert round(summary.mean_nutrition, 2) == 4.0
    assert round(summary.mean_originalite, 2) == round(11 / 3, 2)
    assert summary.mean_coherence == 4.0


def test_load_hitl_ratings_empty_returns_zeros(tmp_path: Path) -> None:
    csv = tmp_path / "hitl.csv"
    csv.write_text("plan_id,nutrition,originalite,coherence\n")

    summary = load_hitl_ratings(csv)

    assert summary.n_ratings == 0
    assert summary.mean_nutrition == 0.0
    assert summary.mean_originalite == 0.0
    assert summary.mean_coherence == 0.0
