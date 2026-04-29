from __future__ import annotations

import pytest

from app.data.recommendations_matrix import (
    GENERIC_FALLBACK,
    RECOMMENDATIONS_MATRIX,
    HealthGoal,
    Imbalance,
    get_recommendation,
)


def test_returns_specific_phrase_for_low_protein_muscle_gain() -> None:
    phrase = get_recommendation(Imbalance.LOW_PROTEIN, HealthGoal.MUSCLE_GAIN)
    assert "prise de masse" in phrase.lower() or "proteine" in phrase.lower()
    assert len(phrase) > 0


def test_returns_generic_fallback_when_health_goal_is_none() -> None:
    phrase = get_recommendation(Imbalance.LOW_PROTEIN, None)
    assert phrase == GENERIC_FALLBACK


def test_matrix_has_all_16_combinations() -> None:
    expected_keys = {
        (imbalance, health_goal)
        for imbalance in Imbalance
        for health_goal in HealthGoal
    }
    assert set(RECOMMENDATIONS_MATRIX.keys()) == expected_keys
    assert len(RECOMMENDATIONS_MATRIX) == 16


@pytest.mark.parametrize("imbalance", list(Imbalance))
@pytest.mark.parametrize("health_goal", list(HealthGoal))
def test_each_combination_returns_non_empty_specific_phrase(
    imbalance: Imbalance, health_goal: HealthGoal
) -> None:
    phrase = get_recommendation(imbalance, health_goal)
    assert phrase != GENERIC_FALLBACK
    assert len(phrase) >= 20


def test_no_em_dashes_in_any_phrase() -> None:
    forbidden = ["—", "–"]
    offenders = [
        (key, phrase)
        for key, phrase in RECOMMENDATIONS_MATRIX.items()
        for char in forbidden
        if char in phrase
    ]
    assert offenders == [], f"phrases contiennent des tirets cadratins : {offenders}"
    for char in forbidden:
        assert char not in GENERIC_FALLBACK
