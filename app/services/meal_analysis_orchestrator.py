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
from app.models.schemas import (
    HealthGoal,
    ImbalanceStatus,
    ImbalanceTag,
    Nutrient,
)
from app.services import llm_client
from app.services.food_classifier import classify_image
from app.services.imbalance_detector import detect, imbalance_to_text
from app.services.nutrition_engine import (
    IncompleteProfile,
    MealType,
    build_user_profile,
)
from app.services.nutrition_lookup import lookup_nutrition

_LOGGER = logging.getLogger(__name__)

_DEFAULT_HEALTH_GOAL = HealthGoal.balance
_CACHE_TTL_DAYS = 30
_MACRO_KEYS = ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")

# Mapping nutrient -> matrix.Imbalance pour le fallback statique. Pas tous les
# nutriments ne sont couverts (matrice 4 cas historique). Quand un nutriment
# n'a pas de pendant matrice on retombe sur le fallback generique.
_NUTRIENT_TO_MATRIX_KEY: dict[tuple[Nutrient, ImbalanceStatus], matrix.Imbalance] = {
    (Nutrient.calories, ImbalanceStatus.excess): matrix.Imbalance.HIGH_CALORIES,
    (Nutrient.protein_g, ImbalanceStatus.deficit): matrix.Imbalance.LOW_PROTEIN,
    (Nutrient.carbs_g, ImbalanceStatus.excess): matrix.Imbalance.HIGH_CARBS,
    (Nutrient.fat_g, ImbalanceStatus.excess): matrix.Imbalance.HIGH_FAT,
}


async def analyze_meal(
    image_bytes: bytes,
    user_id: int,
    db: Session,
    meal_type: MealType | None = None,
) -> dict[str, Any]:
    """Pipeline complet : classification, lookup, tags, LLM synthetique, persistance."""
    # 1. Classification HuggingFace (CPU-bound -> thread pool).
    try:
        predictions = await asyncio.to_thread(classify_image, image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail="Image invalide ou corrompue."
        ) from exc
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

    # 4. Profil + objectif sante.
    goal = (
        db.query(NutritionGoal).filter(NutritionGoal.user_id == user_id).one_or_none()
    )
    health_goal = _resolve_health_goal(goal)
    user_profile = build_user_profile(goal)

    # 5. Profil incomplet : pas de tags, pas de LLM, on indique au front qu'il
    # faut completer la biometrie.
    if isinstance(user_profile, IncompleteProfile):
        analysis = _persist_analysis(
            db=db,
            user_id=user_id,
            detected_foods=detected_foods,
            macros=macros,
            recommendations=[],
            recommendations_hash=None,
            imbalances=[],
            meal_type=meal_type,
        )
        return {
            "analysis_id": analysis.id,
            "detected_foods": detected_foods,
            "macros": macros,
            "imbalances": [],
            "imbalances_text": [],
            "recommendations": [],
            "fallback": False,
            "profile_completion_required": True,
            "missing_fields": list(user_profile.missing_fields),
        }

    # 6. Detection des desequilibres + textes deterministes.
    tags = detect(
        meal_macros=macros,
        profile=goal,
        meal_type=meal_type,
        health_goal=health_goal,
    )
    imbalances_text = [imbalance_to_text(t) for t in tags]

    # 7. Recommandations : cache, sinon LLM unique synthetique.
    if not tags:
        recommendations: list[str] = []
        recommendations_hash: str | None = None
        fallback_used = False
    else:
        recommendations_hash = compute_recommendations_hash(
            top_label, health_goal, tags
        )
        cached = _lookup_cached_recommendations(db, recommendations_hash)
        if cached is not None:
            recommendations = cached
            fallback_used = False
        else:
            recommendations, fallback_used = await _generate_recommendation(
                tags, health_goal, db
            )

    # 8. Persistance. Hash NULL en mode fallback (un appel ulterieur retentera le LLM).
    analysis = _persist_analysis(
        db=db,
        user_id=user_id,
        detected_foods=detected_foods,
        macros=macros,
        recommendations=recommendations,
        recommendations_hash=None if fallback_used else recommendations_hash,
        imbalances=tags,
        meal_type=meal_type,
    )

    return {
        "analysis_id": analysis.id,
        "detected_foods": detected_foods,
        "macros": macros,
        "imbalances": [t.model_dump(mode="json") for t in tags],
        "imbalances_text": imbalances_text,
        "recommendations": recommendations,
        "fallback": fallback_used,
        "profile_completion_required": False,
        "missing_fields": [],
    }


def compute_recommendations_hash(
    top_label: str, health_goal: HealthGoal, tags: list[ImbalanceTag]
) -> str:
    """SHA256 hex du tuple (top_label, health_goal, tags triees)."""
    sorted_tags = sorted(
        ({"nutrient": t.nutrient.value, "status": t.status.value} for t in tags),
        key=lambda d: (d["nutrient"], d["status"]),
    )
    canonical = {
        "top_label": top_label,
        "health_goal": health_goal.value,
        "imbalances": sorted_tags,
    }
    serialized = json.dumps(
        canonical, separators=(",", ":"), sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _resolve_health_goal(goal: NutritionGoal | None) -> HealthGoal:
    if goal is None or not goal.health_goal:
        return _DEFAULT_HEALTH_GOAL
    try:
        return HealthGoal(goal.health_goal)
    except ValueError:
        # CHECK V9 garantit la valeur en BDD ; ce log capture une derive de schema.
        _LOGGER.warning("health_goal inattendu en BDD : %r", goal.health_goal)
        return _DEFAULT_HEALTH_GOAL


def _lookup_cached_recommendations(db: Session, hash_value: str) -> list[str] | None:
    """Cache global TTL 30j : meme hash -> reutilise les recommandations."""
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


async def _generate_recommendation(
    tags: list[ImbalanceTag],
    health_goal: HealthGoal,
    db: Session,
) -> tuple[list[str], bool]:
    """Appelle generate_recommendation (1 seul appel Ollama).

    fallback_used = True si la matrice statique a ete utilisee. La phrase
    retournee est unique : le LLM la synthetise, ou la matrice la concatene.
    """
    fallback_used = False

    def _matrix_fallback(tag_list: list[ImbalanceTag], goal: HealthGoal) -> str:
        nonlocal fallback_used
        fallback_used = True
        try:
            mat_goal = matrix.HealthGoal(goal.value)
        except ValueError:
            return matrix.GENERIC_FALLBACK
        phrases: list[str] = []
        for tag in tag_list:
            mat_imb = _NUTRIENT_TO_MATRIX_KEY.get((tag.nutrient, tag.status))
            if mat_imb is None:
                continue
            phrases.append(matrix.get_recommendation(mat_imb, mat_goal))
        if not phrases:
            return matrix.GENERIC_FALLBACK
        return " ".join(phrases)

    suggestion = await llm_client.generate_recommendation(
        ctx_list=tags,
        health_goal=health_goal,
        db=db,
        fallback=_matrix_fallback,
    )
    if not suggestion:
        return [], fallback_used
    return [suggestion], fallback_used


def _persist_analysis(
    *,
    db: Session,
    user_id: int,
    detected_foods: list[dict[str, Any]],
    macros: dict[str, float],
    recommendations: list[str],
    recommendations_hash: str | None,
    imbalances: list[ImbalanceTag],
    meal_type: MealType | None,
) -> MealAnalysis:
    analysis = MealAnalysis(
        user_id=user_id,
        detected_foods=detected_foods,
        macros=macros,
        confidence_scores={
            item["label"]: item["confidence"] for item in detected_foods
        },
        recommendations=recommendations,
        recommendations_hash=recommendations_hash,
        imbalances=[t.model_dump(mode="json") for t in imbalances],
        meal_type=meal_type.value if meal_type is not None else None,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis
