from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.data import portion_sizes, recommendations_matrix as matrix
from app.db.models import MealAnalysis, NutritionGoal
from app.models.schemas import (
    HealthGoal,
    ImbalanceStatus,
    ImbalanceTag,
    Nutrient,
    ServingSizeLabel,
)
from app.services import llm_client
from app.services.food_classifier import classify_image
from app.services.mistral_vision import classify_image_vision
from app.services.image_thumbnail import to_data_url
from app.services.imbalance_detector import detect, imbalance_to_text
from app.services.nutrition_engine import (
    IncompleteProfile,
    MealType,
    build_user_profile,
)
from app.services.nutrition_lookup import lookup_nutrition
from app.services.user_preferences_service import get_preferences

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


async def _classify(image_bytes: bytes) -> list[tuple[str, float]]:
    """Classifie l'image via le backend configure, avec repli sur Food-101.

    mistral_vision : meilleure reconnaissance et multi-aliments, contrainte au
    catalogue Food-101 (macros toujours ancrees sur la BDD/PNNS). En cas d'echec
    (API indisponible, cle absente, aucun aliment), repli sur le classifieur
    HuggingFace local.
    """
    if settings.analyze_backend == "mistral_vision":
        try:
            preds = await classify_image_vision(image_bytes)
            if preds:
                return preds
            _LOGGER.warning("Mistral vision : aucun aliment detecte, repli Food-101")
        except Exception:
            _LOGGER.exception("Mistral vision indisponible, repli Food-101")
    return await asyncio.to_thread(classify_image, image_bytes)


async def analyze_meal(
    image_bytes: bytes,
    user_id: str,
    db: Session,
    meal_type: MealType | None = None,
) -> dict[str, Any]:
    """Pipeline complet : classification, lookup, tags, LLM synthetique, persistance."""
    # 1. Classification : Mistral vision (contrainte au catalogue) ou Food-101
    #    selon settings.analyze_backend, avec repli sur Food-101.
    try:
        predictions = await _classify(image_bytes)
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail="Image invalide ou corrompue."
        ) from exc
    if not predictions:
        raise HTTPException(
            status_code=422,
            detail=(
                "Aucun aliment reconnu sur cette photo. Le modele identifie des "
                "plats prepares (catalogue Food-101) : cadre le plat de pres, "
                "bien eclaire, et reessaie."
            ),
        )

    # 2. Lookup nutrition pour chaque aliment detecte. Le marqueur source=static
    # (referentiel Food-101 embarque) est retire de la reponse mais alimente un
    # warning de transparence sur l'origine des valeurs.
    detected_foods = []
    estimated_labels: list[str] = []
    missing_labels: list[str] = []
    for label, score in predictions:
        nutrition = lookup_nutrition(label, db)
        if nutrition is None:
            missing_labels.append(label)
        elif nutrition.pop("source", None) == "static":
            estimated_labels.append(label)
        detected_foods.append(
            {"label": label, "confidence": score, "nutrition": nutrition}
        )

    # 3. Tailles de portion PNNS + macros recalculees pour chaque aliment.
    serving_sizes_by_food = [
        _serving_sizes_for(item["label"], item["nutrition"]) for item in detected_foods
    ]

    # 4. Macros du repas : portion medium du premier aliment qui possede des
    # valeurs nutritionnelles. Avant : strictement le top-1, qui pouvait etre
    # sans nutrition et laissait des macros vides alors qu'un autre aliment
    # detecte en avait.
    top_label = predictions[0][0]
    macros_index, macros = _select_meal_macros(serving_sizes_by_food)

    # 5. Profil + objectif sante.
    goal = (
        db.query(NutritionGoal).filter(NutritionGoal.user_id == user_id).one_or_none()
    )
    health_goal = _resolve_health_goal(goal)
    user_profile = build_user_profile(goal)

    warnings = _build_warnings(meal_type, estimated_labels, missing_labels)
    if macros_index != 0:
        warnings.append(
            f"Macros calculees sur {detected_foods[macros_index]['label']} "
            f"(pas de valeurs nutritionnelles pour {top_label})"
        )

    # Thumbnail data URL pour affichage dans l'historique. Calcul une fois,
    # reutilisee dans les deux branches de persistance.
    photo_data_url = to_data_url(image_bytes)

    # 6. Profil incomplet : pas de tags, pas de LLM, on indique au front qu'il
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
            serving_sizes=serving_sizes_by_food,
            photo_url=photo_data_url,
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
            "serving_sizes": serving_sizes_by_food,
            "warnings": warnings,
        }

    # 7. Detection des desequilibres + textes deterministes.
    tags = detect(
        meal_macros=macros,
        profile=goal,
        meal_type=meal_type,
        health_goal=health_goal,
    )
    imbalances_text = [imbalance_to_text(t) for t in tags]

    # 8. Recommandations : cache, sinon LLM unique synthetique.
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
                tags, health_goal, db, user_id
            )

    # 9. Persistance. Hash NULL en mode fallback (un appel ulterieur retentera le LLM).
    analysis = _persist_analysis(
        db=db,
        user_id=user_id,
        detected_foods=detected_foods,
        macros=macros,
        recommendations=recommendations,
        recommendations_hash=None if fallback_used else recommendations_hash,
        imbalances=tags,
        meal_type=meal_type,
        serving_sizes=serving_sizes_by_food,
        photo_url=photo_data_url,
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
        "serving_sizes": serving_sizes_by_food,
        "warnings": warnings,
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
    user_id: str | None = None,
) -> tuple[list[str], bool]:
    """Appelle generate_recommendation (1 seul appel LLM, chain multi-provider).

    Le backend primaire respecte la preference utilisateur, comme les plans
    repas. fallback_used = True si la matrice statique a ete utilisee. La
    phrase retournee est unique : le LLM la synthetise, ou la matrice la
    concatene.
    """
    fallback_used = False
    primary_backend: str | None = None
    if user_id is not None:
        primary_backend = get_preferences(user_id, db).effective_llm.value

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
        primary_backend=primary_backend,
    )
    if not suggestion:
        return [], fallback_used
    return [suggestion], fallback_used


def _persist_analysis(
    *,
    db: Session,
    user_id: str,
    detected_foods: list[dict[str, Any]],
    macros: dict[str, float],
    recommendations: list[str],
    recommendations_hash: str | None,
    imbalances: list[ImbalanceTag],
    meal_type: MealType | None,
    serving_sizes: list[list[dict[str, Any]]],
    photo_url: str | None = None,
) -> MealAnalysis:
    analysis = MealAnalysis(
        user_id=user_id,
        photo_url=photo_url,
        detected_foods=detected_foods,
        macros=macros,
        confidence_scores={
            item["label"]: item["confidence"] for item in detected_foods
        },
        recommendations=recommendations,
        recommendations_hash=recommendations_hash,
        imbalances=[t.model_dump(mode="json") for t in imbalances],
        serving_sizes=serving_sizes,
        meal_type=meal_type.value if meal_type is not None else None,
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)
    return analysis


_MEAL_TYPE_FALLBACK_WARNING = "meal_type non specifie, fallback TDEE/4"


def _select_meal_macros(
    serving_sizes_by_food: list[list[dict]],
) -> tuple[int, dict[str, float]]:
    """Index et macros (portion medium) du premier aliment qui en possede."""
    for i, portions in enumerate(serving_sizes_by_food):
        macros = _medium_portion_macros(portions)
        if macros:
            return i, macros
    return 0, {}


def _build_warnings(
    meal_type: MealType | None,
    estimated_labels: list[str] | None = None,
    missing_labels: list[str] | None = None,
) -> list[str]:
    warnings: list[str] = []
    if meal_type is None:
        warnings.append(_MEAL_TYPE_FALLBACK_WARNING)
    if estimated_labels:
        warnings.append(
            "Valeurs nutritionnelles estimees (referentiel generique) pour : "
            + ", ".join(estimated_labels)
        )
    if missing_labels:
        warnings.append(
            "Valeurs nutritionnelles indisponibles pour : " + ", ".join(missing_labels)
        )
    return warnings


def _serving_sizes_for(
    label: str, nutrition: dict[str, Any] | None
) -> list[dict[str, Any]]:
    """Resout les portions PNNS et y joint les macros recalculees au prorata."""
    portions = portion_sizes.get_serving_sizes(label)
    return [
        {
            "label": p.label.value,
            "grams": p.grams,
            "description": p.description,
            "macros": _scale_macros(nutrition, p.grams),
        }
        for p in portions
    ]


def _scale_macros(nutrition: dict[str, Any] | None, grams: int) -> dict[str, float]:
    """nutrition_entries stocke les valeurs pour 100 g (convention OFF/USDA)."""
    if not nutrition:
        return {}
    factor = grams / 100.0
    return {
        k: float(nutrition[k]) * factor
        for k in _MACRO_KEYS
        if nutrition.get(k) is not None
    }


def _medium_portion_macros(portions: list[dict[str, Any]]) -> dict[str, float]:
    """Renvoie les macros de la portion medium ; vide si aucun macro lookup."""
    for p in portions:
        if p["label"] == ServingSizeLabel.medium.value:
            return dict(p["macros"])
    return {}
