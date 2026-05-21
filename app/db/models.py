from datetime import datetime, timezone

from sqlalchemy import BigInteger, Column, DateTime, Integer, Numeric, String, Text, text
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
    # Tags structures {nutrient, status, delta_pct, target_value, actual_value, unit} (V11).
    # Liste vide quand le profil utilisateur est incomplet (TDEE non calculable).
    imbalances = Column(JSONB)
    # 3 portions (small/medium/large) par item detecte avec macros recalculees (V11).
    serving_sizes = Column(JSONB)
    # breakfast/lunch/dinner/snack ; NULL = fallback TDEE/4 (V11).
    meal_type = Column(Text)
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
    # Sortie de la boucle DeCRIM-light : full / partial_budget / static_fallback (V11).
    compliance_status = Column(Text, nullable=False, server_default=text("'full'"))
    # Strings explicitant les relachements de contraintes (V11).
    compliance_warnings = Column(ARRAY(Text))
    # Backend LLM qui a effectivement genere le plan (V13). Sert au filtre cache
    # backend-aware et a l'audit. DEFAULT 'ollama' pour les lignes pre-V13.
    llm_backend_used = Column(
        String(20), nullable=False, server_default=text("'ollama'")
    )
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
    # Preference utilisateur pour le backend LLM (V13). NULL = utiliser le
    # defaut env (settings.default_llm). Valeurs autorisees au niveau BDD via
    # CHECK ('ollama', 'mistral').
    preferred_llm = Column(String(20))
