from __future__ import annotations

import pytest

from app.data.portion_sizes import (
    _FOOD101_TO_PNNS_CATEGORY,
    _PNNS_SERVING_SIZES,
    get_serving_sizes,
)
from app.models.schemas import ServingSize


def test_known_label_returns_three_portions_small_medium_large() -> None:
    portions = get_serving_sizes("pizza")

    assert len(portions) == 3
    assert {p.label for p in portions} == {"small", "medium", "large"}
    assert all(isinstance(p, ServingSize) for p in portions)
    assert all(p.grams > 0 for p in portions)


def test_unknown_label_returns_medium_100g_fallback() -> None:
    portions = get_serving_sizes("not_a_real_food_label")

    assert len(portions) == 1
    assert portions[0].label == "medium"
    assert portions[0].grams == 100


@pytest.mark.parametrize(
    "category",
    [
        "legumes",
        "feculents",
        "viandes",
        "poissons",
        "fromages",
        "fruits",
        "oleagineux",
        "plats_composes",
        "desserts",
        "boissons",
    ],
)
def test_each_pnns_category_has_three_ordered_portions(category: str) -> None:
    portions = _PNNS_SERVING_SIZES[category]

    assert [p.label for p in portions] == ["small", "medium", "large"]
    assert portions[0].grams < portions[1].grams < portions[2].grams


def test_mapping_has_exactly_101_entries() -> None:
    assert len(_FOOD101_TO_PNNS_CATEGORY) == 101


def test_mapping_only_uses_declared_pnns_categories() -> None:
    declared = set(_PNNS_SERVING_SIZES.keys())
    used = set(_FOOD101_TO_PNNS_CATEGORY.values())

    assert used <= declared, f"unknown categories: {used - declared}"


# Snapshot HITL : tout changement ici doit etre revu manuellement (cf. issue #49).
# Ne JAMAIS regenerer automatiquement ce dict, c'est un garde-fou contre
# les regressions de mapping Food-101 -> PNNS.
EXPECTED_MAPPING: dict[str, str] = {
    "apple_pie": "desserts",
    "baby_back_ribs": "viandes",
    "baklava": "desserts",
    "beef_carpaccio": "viandes",
    "beef_tartare": "viandes",
    "beet_salad": "legumes",
    "beignets": "desserts",
    "bibimbap": "plats_composes",
    "bread_pudding": "desserts",
    "breakfast_burrito": "plats_composes",
    "bruschetta": "plats_composes",
    "caesar_salad": "legumes",
    "cannoli": "desserts",
    "caprese_salad": "legumes",
    "carrot_cake": "desserts",
    "ceviche": "poissons",
    "cheese_plate": "fromages",
    "cheesecake": "desserts",
    "chicken_curry": "plats_composes",
    "chicken_quesadilla": "plats_composes",
    "chicken_wings": "viandes",
    "chocolate_cake": "desserts",
    "chocolate_mousse": "desserts",
    "churros": "desserts",
    "clam_chowder": "plats_composes",
    "club_sandwich": "plats_composes",
    "crab_cakes": "poissons",
    "creme_brulee": "desserts",
    "croque_madame": "plats_composes",
    "cup_cakes": "desserts",
    "deviled_eggs": "plats_composes",
    "donuts": "desserts",
    "dumplings": "plats_composes",
    "edamame": "legumes",
    "eggs_benedict": "plats_composes",
    "escargots": "plats_composes",
    "falafel": "plats_composes",
    "filet_mignon": "viandes",
    "fish_and_chips": "plats_composes",
    "foie_gras": "viandes",
    "french_fries": "feculents",
    "french_onion_soup": "plats_composes",
    "french_toast": "plats_composes",
    "fried_calamari": "poissons",
    "fried_rice": "feculents",
    "frozen_yogurt": "desserts",
    "garlic_bread": "feculents",
    "gnocchi": "feculents",
    "greek_salad": "legumes",
    "grilled_cheese_sandwich": "plats_composes",
    "grilled_salmon": "poissons",
    "guacamole": "legumes",
    "gyoza": "plats_composes",
    "hamburger": "plats_composes",
    "hot_and_sour_soup": "plats_composes",
    "hot_dog": "plats_composes",
    "huevos_rancheros": "plats_composes",
    "hummus": "legumes",
    "ice_cream": "desserts",
    "lasagna": "plats_composes",
    "lobster_bisque": "plats_composes",
    "lobster_roll_sandwich": "plats_composes",
    "macaroni_and_cheese": "plats_composes",
    "macarons": "desserts",
    "miso_soup": "plats_composes",
    "mussels": "poissons",
    "nachos": "plats_composes",
    "omelette": "plats_composes",
    "onion_rings": "legumes",
    "oysters": "poissons",
    "pad_thai": "plats_composes",
    "paella": "plats_composes",
    "pancakes": "desserts",
    "panna_cotta": "desserts",
    "peking_duck": "viandes",
    "pho": "plats_composes",
    "pizza": "plats_composes",
    "pork_chop": "viandes",
    "poutine": "plats_composes",
    "prime_rib": "viandes",
    "pulled_pork_sandwich": "plats_composes",
    "ramen": "plats_composes",
    "ravioli": "plats_composes",
    "red_velvet_cake": "desserts",
    "risotto": "feculents",
    "samosa": "plats_composes",
    "sashimi": "poissons",
    "scallops": "poissons",
    "seaweed_salad": "legumes",
    "shrimp_and_grits": "plats_composes",
    "spaghetti_bolognese": "plats_composes",
    "spaghetti_carbonara": "plats_composes",
    "spring_rolls": "plats_composes",
    "steak": "viandes",
    "strawberry_shortcake": "desserts",
    "sushi": "plats_composes",
    "tacos": "plats_composes",
    "takoyaki": "plats_composes",
    "tiramisu": "desserts",
    "tuna_tartare": "poissons",
    "waffles": "desserts",
}


def test_food101_mapping_matches_validated_snapshot() -> None:
    assert _FOOD101_TO_PNNS_CATEGORY == EXPECTED_MAPPING


def test_known_steak_label_returns_viandes_portions() -> None:
    portions = get_serving_sizes("steak")

    grams_by_label = {p.label.value: p.grams for p in portions}
    assert grams_by_label == {"small": 70, "medium": 100, "large": 150}
