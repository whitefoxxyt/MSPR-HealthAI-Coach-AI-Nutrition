from __future__ import annotations

from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.schemas import PaginatedAnalysesResponse
from app.services import jwt_decoder, meal_analysis_history
from app.services.meal_analysis_orchestrator import analyze_meal as orchestrate

router = APIRouter()

_MAX_SIZE = 10 * 1024 * 1024  # 10 Mo
_ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _user_id_from_auth(authorization: str | None) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authorization header invalide.")
    identity = jwt_decoder.decode(authorization.removeprefix("Bearer "))
    try:
        return int(identity.user_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=401, detail="Sujet JWT invalide.") from exc


_ANALYZE_DESCRIPTION = """
Analyse une photo de repas et renvoie macros, desequilibres et recommandations.

Pipeline applique :

1. Classification par le modele HuggingFace `nateraw/food` (Food-101, top-3 aliments + confiance).
2. Lookup nutritionnel sur la table `nutrition_entries` (datasets ETL).
3. Detection des desequilibres macros (calories, proteines, glucides, lipides) par rapport au profil utilisateur (`nutrition-goals/me`). Si aucun profil n'est configure, l'analyse renvoie quand meme les macros sans recommandations.
4. Generation des recommandations par Ollama (Gemma3:4b) avec cache 30 jours par hash (top_label, health_goal, imbalances). En cas d'echec LLM, repli sur la matrice statique 16 cas (`fallback: true`).
5. Persistance dans `meal_analyses` pour consultation via `GET /api/v1/meal-analyses/me`.

**Limitations** : modele entraine sur Food-101 (101 classes occidentales), inference CPU (1-3 s par appel). Pour les recommandations, le LLM peut etre lent (5-15 s sur CPU), prevoir un timeout cote client.

**Authentification** : header `Authorization: Bearer <jwt>` requis (JWT signe par MSPR-AUTH).
"""

_ANALYZE_SUCCESS_EXAMPLE = {
    "analysis_id": 137,
    "detected_foods": [
        {
            "label": "pizza",
            "confidence": 0.85,
            "nutrition": {
                "calories": 1300,
                "protein_g": 30,
                "carbs_g": 160,
                "fat_g": 50,
                "fiber_g": 5,
            },
        },
        {"label": "lasagna", "confidence": 0.07, "nutrition": None},
    ],
    "macros": {
        "calories": 1300,
        "protein_g": 30,
        "carbs_g": 160,
        "fat_g": 50,
        "fiber_g": 5,
    },
    "imbalances": [
        "Apport calorique eleve (65 % du daily target).",
        "Glucides au-dessus de la cible journaliere.",
    ],
    "recommendations": [
        "Reduis la portion au prochain repas et privilegie une assiette plus protee.",
        "Equilibre la journee avec un diner riche en legumes verts.",
    ],
    "fallback": False,
}


_HISTORY_DESCRIPTION = """
Renvoie l'historique pagine des analyses de repas pour l'utilisateur authentifie.

Tri decroissant par date d'analyse. Chaque element contient les aliments detectes, les macros, les recommandations sauvegardees et la date.

Pagination par `limit` (defaut 20, max 100) et `offset` (defaut 0). Le total est inclus dans la reponse pour permettre au client d'afficher un compteur ou de preparer une pagination infinie.

**Authentification** : header `Authorization: Bearer <jwt>` requis. L'historique est strictement scope a l'utilisateur du JWT.
"""

_HISTORY_RESPONSE_EXAMPLE = {
    "items": [
        {
            "id": 137,
            "detected_foods": [
                {
                    "label": "pizza",
                    "confidence": 0.85,
                    "nutrition": {
                        "calories": 1300,
                        "protein_g": 30,
                        "carbs_g": 160,
                        "fat_g": 50,
                    },
                }
            ],
            "macros": {
                "calories": 1300,
                "protein_g": 30,
                "carbs_g": 160,
                "fat_g": 50,
                "fiber_g": 5,
            },
            "recommendations": ["Reduis la portion au prochain repas."],
            "created_at": "2026-04-29T10:15:00",
        }
    ],
    "total": 137,
    "limit": 20,
    "offset": 0,
}


@router.post(
    "/analyze-meal",
    tags=["Analyse"],
    summary="Analyse une photo de repas (macros + recommandations IA)",
    description=_ANALYZE_DESCRIPTION,
    responses={
        200: {
            "description": "Analyse reussie : macros, desequilibres et recommandations.",
            "content": {"application/json": {"example": _ANALYZE_SUCCESS_EXAMPLE}},
        },
        401: {"description": "JWT manquant, malforme ou invalide."},
        413: {"description": "Image trop volumineuse (limite 10 Mo)."},
        415: {"description": "Type MIME non supporte (utiliser JPEG, PNG ou WebP)."},
        422: {"description": "Image illisible, corrompue ou aucun aliment detecte."},
    },
)
async def analyze_meal(
    photo: UploadFile = File(
        ..., description="Photo du repas (JPEG, PNG ou WebP, 10 Mo max)."
    ),
    authorization: str | None = Header(default=None),
    db: Session = Depends(get_db),
):
    if photo.content_type not in _ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Type non supporte : {photo.content_type}. Utilisez JPEG, PNG ou WebP.",
        )
    image_bytes = await photo.read()
    if len(image_bytes) > _MAX_SIZE:
        raise HTTPException(
            status_code=413, detail="Image trop volumineuse (max 10 Mo)."
        )

    user_id = _user_id_from_auth(authorization)
    return await orchestrate(image_bytes, user_id, db)


@router.get(
    "/meal-analyses/me",
    response_model=PaginatedAnalysesResponse,
    tags=["Historique"],
    summary="Historique pagine des analyses de l'utilisateur",
    description=_HISTORY_DESCRIPTION,
    responses={
        200: {
            "description": "Liste pagine d'analyses (tri decroissant sur created_at).",
            "content": {"application/json": {"example": _HISTORY_RESPONSE_EXAMPLE}},
        },
        401: {"description": "JWT manquant, malforme ou invalide."},
    },
)
def list_my_meal_analyses(
    authorization: str | None = Header(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> PaginatedAnalysesResponse:
    user_id = _user_id_from_auth(authorization)
    return meal_analysis_history.list_user_analyses(user_id, limit, offset, db)
