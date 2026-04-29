from fastapi import FastAPI
from slowapi.errors import RateLimitExceeded
from slowapi.extension import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware

from app.limiter import limiter
from app.routers import health, meal_analysis, meal_plan, nutrition_goals

app = FastAPI(
    title="MSPR HealthAI Coach — AI Nutrition",
    description="Micro-service d'analyse nutritionnelle par IA",
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

app.include_router(health.router)
app.include_router(meal_analysis.router, prefix="/api/v1")
app.include_router(meal_plan.router, prefix="/api/v1")
app.include_router(nutrition_goals.router, prefix="/api/v1")
