from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.models.schemas import FallbackMealPlan


@dataclass(frozen=True)
class FewShotExample:
    """Exemple statique inclus dans _PLAN_PROMPT_TEMPLATE des le 1er essai.

    is_valid=True : exemple positif, plan a imiter (contraintes respectees).
    is_valid=False : exemple negatif, rejection_reason explicite la violation.
    """

    label: str
    plan: FallbackMealPlan
    is_valid: bool
    rejection_reason: str | None = None


_OMNIVORE_WEIGHT_LOSS_PLAN = FallbackMealPlan.model_validate(
    {
        "fallback": False,
        "days": [
            {
                "day": d,
                "meals": [
                    {
                        "name": "Bol avoine fruits rouges",
                        "macros": {
                            "calories": 350,
                            "protein_g": 18.0,
                            "carbs_g": 50.0,
                            "fat_g": 8.0,
                        },
                        "ingredients": [
                            "flocons d'avoine",
                            "lait demi-ecreme",
                            "fruits rouges",
                        ],
                        "est_budget_eur": 1.5,
                        "prep_time_min": 5,
                    },
                    {
                        "name": "Salade poulet quinoa",
                        "macros": {
                            "calories": 480,
                            "protein_g": 38.0,
                            "carbs_g": 45.0,
                            "fat_g": 14.0,
                        },
                        "ingredients": [
                            "blanc de poulet",
                            "quinoa",
                            "salade verte",
                            "tomate",
                        ],
                        "est_budget_eur": 3.5,
                        "prep_time_min": 20,
                    },
                    {
                        "name": "Cabillaud legumes vapeur",
                        "macros": {
                            "calories": 420,
                            "protein_g": 35.0,
                            "carbs_g": 30.0,
                            "fat_g": 12.0,
                        },
                        "ingredients": [
                            "cabillaud",
                            "brocolis",
                            "carottes",
                            "huile d'olive",
                        ],
                        "est_budget_eur": 4.0,
                        "prep_time_min": 25,
                    },
                ],
            }
            for d in range(1, 8)
        ],
    }
)


_VEGAN_GLUTEN_FREE_PLAN = FallbackMealPlan.model_validate(
    {
        "fallback": False,
        "days": [
            {
                "day": d,
                "meals": [
                    {
                        "name": "Porridge sarrasin banane",
                        "macros": {
                            "calories": 380,
                            "protein_g": 12.0,
                            "carbs_g": 60.0,
                            "fat_g": 10.0,
                        },
                        "ingredients": [
                            "flocons de sarrasin",
                            "boisson amande",
                            "banane",
                            "amandes",
                        ],
                        "est_budget_eur": 1.8,
                        "prep_time_min": 8,
                    },
                    {
                        "name": "Bowl pois chiches riz",
                        "macros": {
                            "calories": 520,
                            "protein_g": 22.0,
                            "carbs_g": 70.0,
                            "fat_g": 14.0,
                        },
                        "ingredients": [
                            "pois chiches",
                            "riz basmati",
                            "epinards",
                            "tahini",
                        ],
                        "est_budget_eur": 2.5,
                        "prep_time_min": 25,
                    },
                    {
                        "name": "Curry lentilles corail",
                        "macros": {
                            "calories": 460,
                            "protein_g": 20.0,
                            "carbs_g": 60.0,
                            "fat_g": 12.0,
                        },
                        "ingredients": [
                            "lentilles corail",
                            "huile de coco",
                            "tomate",
                            "epices curry",
                        ],
                        "est_budget_eur": 2.2,
                        "prep_time_min": 25,
                    },
                ],
            }
            for d in range(1, 6)
        ],
    }
)


_NEGATIVE_REJECTED_PLAN = FallbackMealPlan.model_validate(
    {
        "fallback": False,
        "days": [
            {
                "day": 1,
                "meals": [
                    {
                        "name": "Yaourt grec miel",
                        "macros": {
                            "calories": 280,
                            "protein_g": 14.0,
                            "carbs_g": 30.0,
                            "fat_g": 10.0,
                        },
                        "ingredients": [
                            "yaourt grec",
                            "lait entier",
                            "miel",
                            "noix",
                        ],
                        "est_budget_eur": 2.0,
                        "prep_time_min": 5,
                    },
                ],
            }
        ],
    }
)


FEW_SHOT_EXAMPLES: Final[tuple[FewShotExample, ...]] = (
    FewShotExample(
        label="omnivore weight_loss 7 jours",
        plan=_OMNIVORE_WEIGHT_LOSS_PLAN,
        is_valid=True,
    ),
    FewShotExample(
        label="vegan + sans gluten 5 jours",
        plan=_VEGAN_GLUTEN_FREE_PLAN,
        is_valid=True,
    ),
    FewShotExample(
        label="omnivore equilibre, plan rejete",
        plan=_NEGATIVE_REJECTED_PLAN,
        is_valid=False,
        rejection_reason=(
            "Ce plan est rejete car il contient du lait alors que l'utilisateur "
            "est allergique aux laitiers."
        ),
    ),
)
