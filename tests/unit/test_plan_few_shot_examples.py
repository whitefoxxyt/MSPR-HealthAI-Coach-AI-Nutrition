from __future__ import annotations

from app.data.plan_few_shot_examples import FEW_SHOT_EXAMPLES
from app.models.schemas import FallbackMealPlan
from app.services.constraint_validator import ConstraintSpec, validate


# T1 : tracer bullet. 3 entrees, chaque plan parse en FallbackMealPlan.


def test_few_shot_exposes_three_examples_with_valid_plans() -> None:
    assert len(FEW_SHOT_EXAMPLES) == 3
    for example in FEW_SHOT_EXAMPLES:
        assert isinstance(example.plan, FallbackMealPlan)


# T2 : 1ere entree = omnivore + weight_loss + 7 jours, valide pour le validateur.


def test_first_example_is_omnivore_weight_loss_seven_days() -> None:
    example = FEW_SHOT_EXAMPLES[0]
    assert example.is_valid is True
    assert "omnivore" in example.label.lower()
    assert "weight_loss" in example.label.lower()
    assert len(example.plan.days) == 7
    # Validateur cote omnivore : pas de regime banni, pas d'allergie -> 0 violation.
    violations = validate(example.plan, ConstraintSpec(diet_type="omnivore"))
    assert violations == []


# T3 : 2eme entree = vegan + sans gluten, 5 jours, multi-contraintes simultanees.


def test_second_example_respects_vegan_and_gluten_free_simultaneously() -> None:
    example = FEW_SHOT_EXAMPLES[1]
    assert example.is_valid is True
    assert "vegan" in example.label.lower()
    assert "gluten" in example.label.lower()
    assert len(example.plan.days) == 5
    # Le plan doit etre vegan ET sans-gluten en meme temps. Le validateur prend
    # un regime a la fois, on verifie chaque regime separement.
    assert validate(example.plan, ConstraintSpec(diet_type="vegan")) == []
    assert validate(example.plan, ConstraintSpec(diet_type="sans_gluten")) == []


# T4 : 3eme entree = plan negatif annote (allergie laitiers).


def test_third_example_is_negative_with_dairy_allergy_annotation() -> None:
    example = FEW_SHOT_EXAMPLES[2]
    assert example.is_valid is False
    assert example.rejection_reason is not None
    assert "lait" in example.rejection_reason.lower()
    # Le plan negatif doit reellement violer la contrainte annoncee : un allergene
    # 'lait' doit etre detecte par le validateur sur le plan tel quel.
    violations = validate(example.plan, ConstraintSpec(allergies=["lait"]))
    assert any(v.type.value == "allergy" for v in violations)
