from __future__ import annotations

import asyncio

import httpx
from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.db.models import MealAnalysis, NutritionGoal
from app.db.session import get_db
from app.services.food_classifier import classify_image
from app.services.nutrition_lookup import lookup_nutrition
from app.services.spring_client import get_user_me

router = APIRouter()

_MAX_SIZE = 10 * 1024 * 1024  # 10 Mo
_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_imbalances(macros: dict, goal: NutritionGoal | None) -> list[str]:
    if goal is None or not macros:
        return []
    issues = []
    if goal.calories_target and macros.get("calories"):
        ratio = macros["calories"] / float(goal.calories_target)
        if ratio > 0.6:
            issues.append(
                f"Apport calorique élevé : {macros['calories']:.0f} kcal "
                f"({ratio:.0%} de l'objectif journalier de {goal.calories_target} kcal)"
            )
    if goal.protein_g and macros.get("protein_g") is not None:
        if macros["protein_g"] < float(goal.protein_g) * 0.2:
            issues.append(
                f"Faible en protéines : {macros['protein_g']:.1f}g "
                f"(objectif journalier : {goal.protein_g}g)"
            )
    if goal.carbs_g and macros.get("carbs_g") is not None:
        if macros["carbs_g"] > float(goal.carbs_g) * 0.7:
            issues.append(
                f"Élevé en glucides : {macros['carbs_g']:.1f}g "
                f"(objectif journalier : {goal.carbs_g}g)"
            )
    if goal.fat_g and macros.get("fat_g") is not None:
        if macros["fat_g"] > float(goal.fat_g) * 0.7:
            issues.append(
                f"Élevé en lipides : {macros['fat_g']:.1f}g "
                f"(objectif journalier : {goal.fat_g}g)"
            )
    return issues


def _build_recommendations(imbalances: list[str]) -> list[str]:
    if not imbalances:
        return ["Repas équilibré par rapport à vos objectifs nutritionnels."]
    recs = []
    for msg in imbalances:
        if "protéines" in msg:
            recs.append("Ajoutez une source de protéines (poulet, légumineuses, œufs).")
        elif "glucides" in msg:
            recs.append("Réduisez les glucides rapides et préférez les céréales complètes.")
        elif "lipides" in msg:
            recs.append("Limitez les graisses saturées ; privilégiez avocat, noix, huile d'olive.")
        elif "calorique" in msg:
            recs.append("Repas dense en calories : allégez les autres repas de la journée.")
    return recs or ["Consultez un nutritionniste pour des recommandations personnalisées."]


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/analyze-meal")
async def analyze_meal(
    photo: UploadFile = File(...),
    authorization: str = Header(...),
    db: Session = Depends(get_db),
):
    # --- Validation de l'image ---
    if photo.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Type non supporté : {photo.content_type}. Utilisez JPEG, PNG ou WebP.",
        )
    image_bytes = await photo.read()
    if len(image_bytes) > _MAX_SIZE:
        raise HTTPException(status_code=413, detail="Image trop volumineuse (max 10 Mo).")

    # --- Extraction du JWT ---
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    jwt_token = authorization.removeprefix("Bearer ")

    # --- Profil utilisateur via Spring Boot ---
    try:
        user_profile = await get_user_me(jwt_token)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=exc.response.status_code,
            detail="Authentification refusée par le service utilisateur.",
        )
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Service utilisateur indisponible.")

    try:
        user_id: int = user_profile["id"]
    except KeyError:
        raise HTTPException(status_code=502, detail="Réponse inattendue du service utilisateur (champ 'id' manquant).")

    # --- Classification (CPU-bound → thread pool) ---
    try:
        predictions = await asyncio.to_thread(classify_image, image_bytes)
    except Exception:
        raise HTTPException(status_code=422, detail="Image invalide ou corrompue.")
    if not predictions:
        raise HTTPException(status_code=422, detail="Aucun aliment détecté avec un score suffisant.")

    # --- Lookup nutritionnel pour chaque aliment détecté ---
    detected_foods = [
        {"label": label, "confidence": score, "nutrition": lookup_nutrition(label, db)}
        for label, score in predictions
    ]

    # --- Macros du repas (aliment le plus probable) ---
    top_nutrition = detected_foods[0]["nutrition"]
    macros: dict = {}
    if top_nutrition:
        macros = {
            k: top_nutrition[k]
            for k in ("calories", "protein_g", "carbs_g", "fat_g", "fiber_g")
            if top_nutrition.get(k) is not None
        }

    # --- Objectifs nutritionnels + déséquilibres ---
    goal = db.query(NutritionGoal).filter(NutritionGoal.user_id == user_id).first()
    imbalances = _detect_imbalances(macros, goal)
    recommendations = _build_recommendations(imbalances)

    # --- Sauvegarde ---
    analysis = MealAnalysis(
        user_id=user_id,
        detected_foods=detected_foods,
        macros=macros,
        confidence_scores={item["label"]: item["confidence"] for item in detected_foods},
    )
    db.add(analysis)
    db.commit()
    db.refresh(analysis)

    return {
        "analysis_id": analysis.id,
        "detected_foods": detected_foods,
        "macros": macros,
        "imbalances": imbalances,
        "recommendations": recommendations,
    }
