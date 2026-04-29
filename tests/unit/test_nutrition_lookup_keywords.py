from __future__ import annotations

from app.services.nutrition_lookup import _label_to_keywords


def test_label_to_keywords_splits_underscores() -> None:
    assert _label_to_keywords("grilled_salmon") == ["grilled", "salmon"]


def test_label_to_keywords_lowercases() -> None:
    assert _label_to_keywords("Beef_Carpaccio") == ["beef", "carpaccio"]


def test_label_to_keywords_single_word() -> None:
    assert _label_to_keywords("pizza") == ["pizza"]
