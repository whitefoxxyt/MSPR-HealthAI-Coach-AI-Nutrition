from fastapi import FastAPI

from app.routers import health, meal_analysis, nutrition_goals

app = FastAPI(
    title="MSPR HealthAI Coach — AI Nutrition",
    description="Micro-service d'analyse nutritionnelle par IA",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(meal_analysis.router, prefix="/api/v1")
app.include_router(nutrition_goals.router, prefix="/api/v1")
