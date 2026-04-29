from __future__ import annotations

from collections.abc import Callable
from io import BytesIO
from typing import Any

from PIL import Image

from app.services.food_classifier import classify_image


def _png_bytes(
    size: tuple[int, int] = (4, 4), color: tuple[int, int, int] = (255, 0, 0)
) -> bytes:
    buf = BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


_PNG_4X4 = _png_bytes()


def test_classify_image_uses_patched_classifier(
    mock_classifier: Callable[..., list[dict[str, Any]]],
) -> None:
    results = classify_image(_PNG_4X4, top_k=2, confidence_threshold=0.0)

    assert results == [("pizza", 0.85), ("lasagna", 0.07)]


def test_confidence_threshold_filters_low_scores(
    mock_classifier: Callable[..., list[dict[str, Any]]],
) -> None:
    results = classify_image(_PNG_4X4, top_k=5, confidence_threshold=0.10)

    assert [label for label, _ in results] == ["pizza"]
