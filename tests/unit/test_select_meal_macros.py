from __future__ import annotations

from app.services.meal_analysis_orchestrator import _select_meal_macros

_MEDIUM_WITH_MACROS = {
    "label": "medium",
    "grams": 100,
    "description": "100 g",
    "macros": {"calories": 200.0, "protein_g": 10.0},
}
_MEDIUM_EMPTY = {"label": "medium", "grams": 100, "description": "100 g", "macros": {}}


def test_prend_le_top1_quand_il_a_des_macros() -> None:
    index, macros = _select_meal_macros([[_MEDIUM_WITH_MACROS], [_MEDIUM_EMPTY]])

    assert index == 0
    assert macros["calories"] == 200.0


def test_bascule_sur_le_premier_aliment_avec_macros() -> None:
    # Top-1 sans valeurs nutritionnelles : avant, les macros du repas
    # restaient vides alors que le 2e aliment detecte en avait.
    index, macros = _select_meal_macros([[_MEDIUM_EMPTY], [_MEDIUM_WITH_MACROS]])

    assert index == 1
    assert macros["protein_g"] == 10.0


def test_aucun_aliment_avec_macros() -> None:
    index, macros = _select_meal_macros([[_MEDIUM_EMPTY], [_MEDIUM_EMPTY]])

    assert index == 0
    assert macros == {}
