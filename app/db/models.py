from datetime import datetime

from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, Numeric, String, text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class MealAnalysis(Base):
    __tablename__ = "meal_analyses"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    photo_url = Column(String(500))
    detected_foods = Column(JSONB, nullable=False, server_default=text("'[]'"))
    macros = Column(JSONB, nullable=False, server_default=text("'{}'"))
    confidence_scores = Column(JSONB, nullable=False, server_default=text("'{}'"))
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class MealPlan(Base):
    __tablename__ = "meal_plans"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    plan = Column(JSONB, nullable=False, server_default=text("'{}'"))
    objective = Column(String(100))
    constraints = Column(JSONB, nullable=False, server_default=text("'{}'"))
    generated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class NutritionGoal(Base):
    __tablename__ = "nutrition_goals"

    user_id = Column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    calories_target = Column(Integer)
    protein_g = Column(Numeric(8, 2))
    carbs_g = Column(Numeric(8, 2))
    fat_g = Column(Numeric(8, 2))
    allergies = Column(ARRAY(String))
    diet_type = Column(String(50))
