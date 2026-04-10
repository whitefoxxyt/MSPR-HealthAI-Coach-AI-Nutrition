from fastapi import FastAPI

from app.routers import health

app = FastAPI(
    title="MSPR HealthAI Coach — AI Nutrition",
    description="Micro-service d'analyse nutritionnelle par IA",
    version="1.0.0",
)

app.include_router(health.router)
