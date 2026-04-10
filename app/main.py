from fastapi import FastAPI

from app.routers import health, meal_analysis

app = FastAPI(
    title="MSPR HealthAI Coach — AI Nutrition",
    description="Micro-service d'analyse nutritionnelle par IA",
    version="1.0.0",
)

app.include_router(health.router)
app.include_router(meal_analysis.router)
