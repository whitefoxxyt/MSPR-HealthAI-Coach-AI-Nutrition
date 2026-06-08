from __future__ import annotations

import json
import logging

import httpx

from app.config import settings
from app.data.portion_sizes import FOOD101_LABELS
from app.services.image_thumbnail import to_data_url

_LOGGER = logging.getLogger(__name__)

_TIMEOUT_S = 60.0
_DEFAULT_CONFIDENCE_THRESHOLD = 0.10

# Sortie JSON attendue. La contrainte au catalogue est rappelee dans le prompt
# ET reverifiee en code (filtrage sur FOOD101_LABELS) : on ne depend pas du mode
# strict sur un enum de 101 valeurs.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "foods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["label", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["foods"],
    "additionalProperties": False,
}


def _build_prompt() -> str:
    catalogue = ", ".join(FOOD101_LABELS)
    return (
        "Tu es un classifieur d'aliments. Regarde la photo et identifie le ou "
        "les aliments presents, en te limitant STRICTEMENT a cette liste de "
        "labels (catalogue Food-101) :\n"
        f"{catalogue}\n"
        "Pour chaque aliment visible, renvoie le label EXACT de la liste le "
        "plus proche et une confiance entre 0 et 1. Renvoie plusieurs entrees "
        "si plusieurs aliments sont presents, triees par confiance decroissante. "
        "N'invente aucun label hors de la liste."
    )


def _normalize(label: str) -> str:
    return label.strip().lower().replace(" ", "_")


async def classify_image_vision(
    image_bytes: bytes,
    top_k: int = 5,
    confidence_threshold: float = _DEFAULT_CONFIDENCE_THRESHOLD,
) -> list[tuple[str, float]]:
    """Classifie une image via Mistral vision, contrainte au catalogue Food-101.

    Meme contrat que food_classifier.classify_image : liste de (label, score)
    triee par score decroissant, filtree par seuil. Les labels hors catalogue
    sont ecartes. Leve une exception en cas d'echec API (l'orchestrateur retombe
    alors sur Food-101).
    """
    if not settings.mistral_api_key:
        raise ValueError("MISTRAL_API_KEY manquante pour la vision Mistral.")
    data_url = to_data_url(image_bytes)
    if data_url is None:
        raise ValueError("Image illisible.")

    payload = {
        "model": settings.mistral_vision_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _build_prompt()},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "food_detection",
                "schema": _RESPONSE_SCHEMA,
                "strict": True,
            },
        },
        "temperature": 0,
    }
    headers = {"Authorization": f"Bearer {settings.mistral_api_key}"}
    base = settings.mistral_base_url.rstrip("/")
    async with httpx.AsyncClient(timeout=_TIMEOUT_S) as client:
        resp = await client.post(
            f"{base}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()

    content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
    parsed = json.loads(content)

    valid = set(FOOD101_LABELS)
    seen: set[str] = set()
    results: list[tuple[str, float]] = []
    for food in parsed.get("foods", []):
        label = _normalize(str(food.get("label", "")))
        if label not in valid or label in seen:
            continue
        try:
            score = round(float(food.get("confidence", 0.0)), 4)
        except (TypeError, ValueError):
            continue
        if score < confidence_threshold:
            continue
        seen.add(label)
        results.append((label, score))

    results.sort(key=lambda t: t[1], reverse=True)
    return results[:top_k]
