from __future__ import annotations

# Verifie que les colonnes ajoutees par MSPR-DB V11 (slice 1 du PRD #45) sont
# exposees par les ORM et persistees correctement (insert -> commit -> re-read).

from sqlalchemy.orm import Session

from app.db.models import MealAnalysis, MealPlan


def test_meal_analysis_persists_v11_fields(db_session: Session) -> None:
    imbalances = [
        {
            "nutrient": "protein",
            "status": "deficit",
            "delta_pct": -25.0,
            "target_value": 30.0,
            "actual_value": 22.5,
            "unit": "g",
        },
    ]
    serving_sizes = [
        {
            "label": "small",
            "grams": 120,
            "macros": {"calories": 240, "protein_g": 12, "carbs_g": 30, "fat_g": 8},
        },
        {
            "label": "medium",
            "grams": 200,
            "macros": {"calories": 400, "protein_g": 20, "carbs_g": 50, "fat_g": 14},
        },
        {
            "label": "large",
            "grams": 300,
            "macros": {"calories": 600, "protein_g": 30, "carbs_g": 75, "fat_g": 20},
        },
    ]

    analysis = MealAnalysis(
        user_id=42,
        photo_url="https://example.test/p.jpg",
        detected_foods=[{"label": "pizza", "score": 0.85}],
        macros={"calories": 400, "protein_g": 20},
        confidence_scores={"pizza": 0.85},
        imbalances=imbalances,
        serving_sizes=serving_sizes,
        meal_type="lunch",
    )
    db_session.add(analysis)
    db_session.commit()

    fetched = db_session.query(MealAnalysis).filter_by(user_id=42).one()
    assert fetched.imbalances == imbalances
    assert fetched.serving_sizes == serving_sizes
    assert fetched.meal_type == "lunch"


def test_meal_analysis_v11_fields_default_to_null(db_session: Session) -> None:
    # Les 3 colonnes V11 sur meal_analyses sont nullable. Quand on insere sans
    # les renseigner, elles doivent rester NULL (pas de default applicatif).
    analysis = MealAnalysis(user_id=7)
    db_session.add(analysis)
    db_session.commit()

    fetched = db_session.query(MealAnalysis).filter_by(user_id=7).one()
    assert fetched.imbalances is None
    assert fetched.serving_sizes is None
    assert fetched.meal_type is None


def test_meal_plan_persists_v11_fields(db_session: Session) -> None:
    warnings = ["Budget journalier depasse de 1.20 EUR le mardi"]

    plan = MealPlan(
        user_id=99,
        plan={"days": []},
        objective="weight_loss",
        constraints={"allergies": ["peanut"], "diet": "vegan", "budget_eur_per_day": 8.0},
        compliance_status="partial_budget",
        compliance_warnings=warnings,
    )
    db_session.add(plan)
    db_session.commit()

    fetched = db_session.query(MealPlan).filter_by(user_id=99).one()
    assert fetched.compliance_status == "partial_budget"
    assert fetched.compliance_warnings == warnings


def test_meal_plan_compliance_status_defaults_to_full(db_session: Session) -> None:
    # V11 : compliance_status TEXT NOT NULL DEFAULT 'full'. Insertion sans la
    # colonne doit retourner 'full' apres flush + refresh (default cote BDD).
    plan = MealPlan(user_id=11)
    db_session.add(plan)
    db_session.commit()
    db_session.refresh(plan)

    assert plan.compliance_status == "full"
    assert plan.compliance_warnings is None
