from __future__ import annotations

from io import BytesIO

from PIL import Image
from transformers import pipeline

_classifier = None
_MODEL_ID = "nateraw/food"
DEFAULT_CONFIDENCE_THRESHOLD = 0.10


def _get_classifier():
    global _classifier
    if _classifier is None:
        _classifier = pipeline("image-classification", model=_MODEL_ID)
    return _classifier


def classify_image(
    image_bytes: bytes,
    top_k: int = 5,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[tuple[str, float]]:
    """
    Classifie une image alimentaire.

    Retourne une liste de (label, score) triée par score décroissant,
    filtrée par confidence_threshold.
    Label : chaîne Food-101 avec underscores (ex: "grilled_salmon").
    """
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    results = _get_classifier()(image, top_k=top_k)
    return [
        (r["label"], round(r["score"], 4))
        for r in results
        if r["score"] >= confidence_threshold
    ]
