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


# Snapshot HITL : SHA256 du mapping canonicalise. Tout changement aux 101 entrees
# de _FOOD101_TO_PNNS_CATEGORY change le hash et casse ce test, forcant une
# revue humaine (cf. issue #49). Pour mettre a jour : recalculer avec
# `python -c "import json,hashlib; from app.data.portion_sizes import \
# _FOOD101_TO_PNNS_CATEGORY as m; print(hashlib.sha256(json.dumps(m, \
# sort_keys=True, ensure_ascii=False).encode()).hexdigest())"`
import hashlib
import json

EXPECTED_MAPPING_SHA256 = (
    "142fdf245825f0a7432adc17f9c8ada39db328d758fbfbbb1b0988c38341c806"
)


def test_food101_mapping_matches_validated_snapshot() -> None:
    canonical = json.dumps(
        _FOOD101_TO_PNNS_CATEGORY, sort_keys=True, ensure_ascii=False
    )
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert actual == EXPECTED_MAPPING_SHA256, (
        f"Mapping Food-101 -> PNNS modifie. Hash attendu : "
        f"{EXPECTED_MAPPING_SHA256}, hash actuel : {actual}. "
        "Revue manuelle requise puis mise a jour de EXPECTED_MAPPING_SHA256."
    )


def test_known_steak_label_returns_viandes_portions() -> None:
    portions = get_serving_sizes("steak")

    grams_by_label = {p.label.value: p.grams for p in portions}
    assert grams_by_label == {"small": 70, "medium": 100, "large": 150}
