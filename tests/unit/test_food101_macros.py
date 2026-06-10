from __future__ import annotations

import pytest

from app.data.food101_macros import FOOD101_MACROS, static_nutrition_for
from app.data.portion_sizes import _FOOD101_TO_PNNS_CATEGORY

_MACRO_KEYS = ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")


def test_couvre_tous_les_labels_food101() -> None:
    # Garantie centrale : chaque classe du modele a des macros de repli,
    # une analyse ne peut plus retourner des macros vides faute de lookup.
    manquants = sorted(set(_FOOD101_TO_PNNS_CATEGORY) - set(FOOD101_MACROS))
    assert manquants == []


def test_entrees_completes_et_bornees() -> None:
    for label, entry in FOOD101_MACROS.items():
        assert entry["food_name"], label
        for key in _MACRO_KEYS:
            value = entry[key]
            assert isinstance(value, float), (label, key)
            assert value >= 0.0, (label, key)
        # Bornes plausibles pour 100 g d'un plat prepare.
        assert 30.0 <= entry["calories"] <= 600.0, label


@pytest.mark.parametrize("label", sorted(FOOD101_MACROS))
def test_coherence_atwater(label: str) -> None:
    # 4 kcal/g proteines et glucides, 9 kcal/g lipides : tolerance 15 %
    # (alcool, sucres-alcools et arrondis non modelises).
    entry = FOOD101_MACROS[label]
    atwater = 4 * entry["protein_g"] + 4 * entry["carbs_g"] + 9 * entry["fat_g"]
    assert atwater == pytest.approx(entry["calories"], rel=0.15), label


def test_static_nutrition_for_marque_la_source() -> None:
    result = static_nutrition_for("greek_salad")

    assert result is not None
    assert result["source"] == "static"
    assert result["food_name"] == "Greek salad"
    # L'original ne doit pas etre mute par l'ajout du marqueur.
    assert "source" not in FOOD101_MACROS["greek_salad"]


def test_static_nutrition_for_label_inconnu() -> None:
    assert static_nutrition_for("plat_inconnu") is None
