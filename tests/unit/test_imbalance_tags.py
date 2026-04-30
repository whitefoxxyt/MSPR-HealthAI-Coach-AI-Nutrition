from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.models.schemas import ImbalanceStatus, ImbalanceTag, Nutrient


def test_imbalance_tag_accepts_valid_excess_calories() -> None:
    tag = ImbalanceTag(
        nutrient=Nutrient.calories,
        status=ImbalanceStatus.excess,
        delta_pct=0.30,
        target_value=689.0,
        actual_value=900.0,
        unit="kcal",
    )

    assert tag.nutrient is Nutrient.calories
    assert tag.status is ImbalanceStatus.excess
    assert tag.delta_pct == pytest.approx(0.30)
    assert tag.unit == "kcal"


def test_imbalance_tag_rejects_unknown_nutrient() -> None:
    with pytest.raises(ValidationError):
        ImbalanceTag(
            nutrient="vitamin_x",  # type: ignore[arg-type]
            status=ImbalanceStatus.excess,
            delta_pct=0.0,
            target_value=10.0,
            actual_value=10.0,
            unit="g",
        )


def test_nutrient_enum_lists_required_six_values() -> None:
    # Garde-fou : l'issue 51 fixe ces 6 nutriments. Si quelqu'un en supprime un
    # par erreur, ce test casse.
    assert {n.value for n in Nutrient} == {
        "calories",
        "protein_g",
        "carbs_g",
        "fat_g",
        "fibers_g",
        "saturated_fat_g",
    }


def test_imbalance_status_enum_lists_three_values() -> None:
    assert {s.value for s in ImbalanceStatus} == {"excess", "deficit", "ok"}
