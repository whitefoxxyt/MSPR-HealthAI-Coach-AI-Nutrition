from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.data import recommendations_matrix as matrix
from app.db.models import MealAnalysis, NutritionGoal
from app.models.schemas import HealthGoal, Imbalance, RecommendationContext
from app.services import llm_client
from app.services.food_classifier import classify_image
from app.services.imbalance_detector import detect_imbalances
from app.services.nutrition_lookup import lookup_nutrition

_LOGGER = logging.getLogger(__name__)

# Mapping schemas.Imbalance -> matrix.Imbalance (4 cas detectables seulement).
# Les 4 cles sont les seules que `imbalance_detector` peut produire.
_IMBALANCE_TO_MATRIX: dict[Imbalance, matrix.Imbalance] = {
    Imbalance.calories_high: matrix.Imbalance.HIGH_CALORIES,
    Imbalance.protein_low: matrix.Imbalance.LOW_PROTEIN,
    Imbalance.carbs_high: matrix.Imbalance.HIGH_CARBS,
    Imbalance.fat_high: matrix.Imbalance.HIGH_FAT,
}

_DEFAULT_HEALTH_GOAL = HealthGoal.balance
_CACHE_TTL_DAYS = 30
_MACRO_KEYS = ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")


async def analyze_meal(image_bytes: bytes, user_id: int, db: Session) -> dict[str, Any]:
    """Pipeline complet : classification, lookup, imbalances, LLM, persistance."""
    # 1. Classification HuggingFace (CPU-bound -> thread pool).
    try:
        predictions = await asyncio.to_thread(classify_image, image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Image invalide ou corrompue.") from exc
    if not predictions:
        raise HTTPException(
            status_code=422, detail="Aucun aliment detecte avec un score suffisant."
        )

    # 2. Lookup nutrition pour chaque aliment detecte.
    detected_foods = [
        {"label": label, "confidence": score, "nutrition": lookup_nutrition(label, db)}
        for label, score in predictions
    ]

    # 3. Macros du repas (aliment le plus probable).
    top_label = predictions[0][0]
    top_nutrition = detected_foods[0]["nutrition"] or {}
    macros: dict[str, float] = {
        k: top_nutrition[k] for k in _MACRO_KEYS if top_nutrition.get(k) is not None
    }

    # 4. Profil + desequilibres + objectif de sante.
    goal = db.query(NutritionGoal).filter(NutritionGoal.user_id == user_id).one_or_none()
    imbalances = detect_imbalances(macros, goal)
    imbalance_messages = [msg for _, msg in imbalances]
    imbalance_kinds = [kind for kind, _ in imbalances]
    health_goal = _resolve_health_goal(goal)

    # 5. Recommandations : cache, sinon LLM avec fallback matrice.
    if not imbalance_kinds:
        recommendations: list[str] = []
        recommendations_hash: str | None = None
        fallback_used = False
    else:
        recommendations_hash = compute_recommendations_hash(
            top_label, health_goal, imbalance_kinds
        )
        cached = _lookup_cached_recommendations(db, recommendations_hash)
        if cached is not None:
            recommendations = cached
            fallback_used = False
        else:
            recommendations, fallback_used = await _generate_recommendations(
                user_id, imbalance_kinds, health_goal, db
            )

    # 6. Persistance. Hash NULL en mode fallback : un appel ulterieur retentera le LLM.
    analysis = MealAnalysis(
        user_id=user_id,
        detected_foods=detected_foods,
        macros=macros,
        confidence_scores={item["label"]: item["confidence"] for item in detected_foods},
        recommendations=recommendations,
        recommendations_hash=None if fallback_used else recommendations_hash,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return {
        "analysis_id": analysis.id,
        "detected_foods": detected_foods,
        "macros": macros,
        "imbalances": imbalance_messages,
        "recommendations": recommendations,
        "fallback": fallback_used,
    }


def compute_recommendations_hash(
    top_label: str, health_goal: HealthGoal, imbalances: list[Imbalance]
) -> str:
    """SHA256 hex de (top_label, health_goal, imbalances triees) en JSON canonique."""
    canonical = {
        "top_label": top_label,
        "health_goal": health_goal.value,
        "imbalances": sorted(i.value for i in imbalances),
    }
    serialized = json.dumps(canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resolve_health_goal(goal: NutritionGoal | None) -> HealthGoal:
    if goal is None or not goal.health_goal:
        return _DEFAULT_HEALTH_GOAL
    try:
        return HealthGoal(goal.health_goal)
    except ValueError:
        # Une valeur non standard en BDD (extremement improbable car CHECK V9) :
        # on degrade vers balance plutot que crasher l'analyse.
        _LOGGER.warning("health_goal inattendu en BDD : %r", goal.health_goal)
        return _DEFAULT_HEALTH_GOAL


def _lookup_cached_recommendations(db: Session, hash_value: str) -> list[str] | None:
    """Cache global : si une analyse < TTL jours partage le meme hash, on la reutilise.

    Race condition assumee : deux requetes concurrentes avec le meme hash peuvent
    declencher chacune un appel LLM avant que la premiere n'ecrive en BDD. La
    seconde ecrasera juste l'entree avec une valeur equivalente. Volume actuel
    trop faible pour justifier un INSERT ON CONFLICT.
    """
    row = db.execute(
        text(
            "SELECT recommendations FROM meal_analyses "
            "WHERE recommendations_hash = :h "
            "AND created_at > NOW() - make_interval(days => :ttl_days) "
            "ORDER BY created_at DESC "
            "LIMIT 1"
        ),
        {"h": hash_value, "ttl_days": _CACHE_TTL_DAYS},
    ).fetchone()
    if row is None:
        return None
    return list(row.recommendations) if row.recommendations else []


async def _generate_recommendations(
    user_id: int,
    imbalances: list[Imbalance],
    health_goal: HealthGoal,
    db: Session,
) -> tuple[list[str], bool]:
    """Appelle generate_recommendation pour chaque desequilibre.

    fallback_used = True si AU MOINS UN appel a echoue et a active la matrice
    statique. La liste retournee peut donc melanger des phrases LLM et matrice
    dans ce cas ; le client n'a pas le detail granulaire.
    """
    fallback_used = False

    def _matrix_fallback(imb: Imbalance, goal: HealthGoal) -> str:
        nonlocal fallback_used
        fallback_used = True
        mat_imb = _IMBALANCE_TO_MATRIX.get(imb)
        if mat_imb is None:
            return matrix.GENERIC_FALLBACK
        try:
            mat_goal = matrix.HealthGoal(goal.value)
        except ValueError:
            return matrix.GENERIC_FALLBACK
        return matrix.get_recommendation(mat_imb, mat_goal)

    async def _one(kind: Imbalance) -> str:
        ctx = RecommendationContext(
            user_id=user_id, imbalance=kind, health_goal=health_goal
        )
        return await llm_client.generate_recommendation(ctx, db, fallback=_matrix_fallback)

    # Parallelise les appels LLM ; le semaphore d'llm_client plafonne deja la
    # charge Ollama (max 2 inferences simultanees).
    recommendations = await asyncio.gather(*(_one(k) for k in imbalances))
    return list(recommendations), fallback_used
