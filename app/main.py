from fastapi import FastAPI
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.limiter import limiter
from app.routers import health, meal_analysis, meal_plan, nutrition_goals

API_DESCRIPTION = """
Micro-service d'analyse nutritionnelle et de generation de plans repas par IA, partie de la plateforme MSPR HealthAI Coach.

Deux pipelines d'IA sont exposes :

- **Analyse de repas** : photo en entree, classification HuggingFace (`nateraw/food`, modele Food-101), lookup nutritionnel sur les datasets ETL, detection de desequilibres macros vs profil utilisateur, puis recommandations generees par Ollama (Gemma3:4b) avec fallback matrice statique.
- **Generation de plans repas** : objectif de sante, preferences alimentaires et budget, prompt structure soumis a Ollama, validation du JSON et persistance du plan sur 1 a 30 jours.

Le service expose egalement la gestion du profil nutritionnel (`nutrition-goals/me`) et l'historique pagine des analyses (`meal-analyses/me`). L'authentification se fait via JWT signe par MSPR-AUTH (`Authorization: Bearer <jwt>`).
"""

TAGS_METADATA = [
    {
        "name": "Analyse",
        "description": "Analyse d'un repas a partir d'une photo : classification HuggingFace, macros et recommandations.",
    },
    {
        "name": "Plans",
        "description": "Generation de plans repas personnalises via Ollama (avec fallback matrice statique).",
    },
    {
        "name": "Profil",
        "description": "Gestion du profil nutritionnel de l'utilisateur (objectifs caloriques, macros cibles, allergies, regime).",
    },
    {
        "name": "Historique",
        "description": "Lecture pagine des analyses passees pour l'utilisateur authentifie.",
    },
    {
        "name": "Sante",
        "description": "Healthcheck du service et de ses dependances (PostgreSQL, Ollama).",
    },
]

app = FastAPI(
    title="MSPR HealthAI Coach - AI Nutrition",
    description=API_DESCRIPTION,
    version="1.0.0",
    contact={
        "name": "Arthur Poncin",
        "email": "arthur.poncin@sixense-group.com",
    },
    openapi_tags=TAGS_METADATA,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(meal_analysis.router, prefix="/api/v1")
app.include_router(meal_plan.router, prefix="/api/v1")
app.include_router(nutrition_goals.router, prefix="/api/v1")
