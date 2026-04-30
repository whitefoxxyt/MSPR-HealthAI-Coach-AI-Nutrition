from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Integer, Numeric, String, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase


# user_id est un identifiant opaque venant du JWT (decode local avec
# BETTER_AUTH_SECRET). Pas de FK : la table users a ete droppee par MSPR-DB
# V7__drop_users_table.sql.
class Base(DeclarativeBase):
    pass


class MealAnalysis(Base):
    __tablename__ = "meal_analyses"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    photo_url = Column(String(500))
    detected_foods = Column(JSONB, nullable=False, server_default=text("'[]'"))
    macros = Column(JSONB, nullable=False, server_default=text("'{}'"))
    confidence_scores = Column(JSONB, nullable=False, server_default=text("'{}'"))
    # Recommandations textuelles : LLM Ollama avec fallback matrice statique (V9).
    recommendations = Column(JSONB, nullable=False, server_default=text("'[]'"))
    # Cle de cache (top_food_label, health_goal, imbalances). 30 jours TTL (V10).
    recommendations_hash = Column(String(64))
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class MealPlan(Base):
    __tablename__ = "meal_plans"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False)
    plan = Column(JSONB, nullable=False, server_default=text("'{}'"))
    objective = Column(String(100))
    constraints = Column(JSONB, nullable=False, server_default=text("'{}'"))
    # SHA256 des inputs canonicalises ; cle de cache (V9). Index dedie en BDD.
    inputs_hash = Column(String(64))
    generated_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))


class NutritionGoal(Base):
    __tablename__ = "nutrition_goals"

    user_id = Column(BigInteger, primary_key=True)
    calories_target = Column(Integer)
    protein_g = Column(Numeric(8, 2))
    carbs_g = Column(Numeric(8, 2))
    fat_g = Column(Numeric(8, 2))
    allergies = Column(ARRAY(String))
    diet_type = Column(String(50))
    health_goal = Column(String(30))
    # Biometrie (V12) pour le calcul TDEE via Mifflin-St Jeor.
    gender = Column(String(10))
    age = Column(Integer)
    weight_kg = Column(Numeric(5, 2))
    height_cm = Column(Numeric(5, 2))
    activity_level = Column(String(20))
