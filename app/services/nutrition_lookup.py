from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.orm import Session


def _label_to_keywords(label: str) -> list[str]:
    """Convertit un label Food-101 (ex: 'grilled_salmon') en mots-clés de recherche."""
    return label.replace("_", " ").lower().split()


# Note : la colonne nutrition_entries.user_id a ete supprimee par MSPR-DB
# V7__drop_users_table.sql (drop de la table users + des FK associees). Les
# requetes ci-dessous ne filtrent donc plus sur user_id.
def lookup_nutrition(
    label: str, db: Session
) -> dict | None:
    """
    Recherche les données nutritionnelles pour un label Food-101.

    Stratégie :
    1. Correspondance exacte (insensible à la casse) après normalisation des underscores.
    2. Correspondance partielle : chaque mot-clé doit apparaître dans food_name.

    Retourne un dict avec les macros, ou None si aucune entrée trouvée.
    """
    normalized = label.replace("_", " ")

    # 1. Exact match
    row = db.execute(
        text(
            "SELECT food_name, calories, protein_g, carbs_g, fat_g, fiber_g "
            "FROM nutrition_entries "
            "WHERE LOWER(food_name) = LOWER(:name) "
            "LIMIT 1"
        ),
        {"name": normalized},
    ).fetchone()

    if row is None:
        # 2. Fuzzy match : tous les mots-clés présents dans food_name
        keywords = _label_to_keywords(label)
        if keywords:
            conditions = " AND ".join(
                f"LOWER(food_name) LIKE :kw{i}" for i in range(len(keywords))
            )
            params = {f"kw{i}": f"%{kw}%" for i, kw in enumerate(keywords)}
            row = db.execute(
                text(
                    "SELECT food_name, calories, protein_g, carbs_g, fat_g, fiber_g "
                    "FROM nutrition_entries "
                    f"WHERE {conditions} "
                    "LIMIT 1"
                ),
                params,
            ).fetchone()

    if row is None:
        return None

    return {
        "food_name": row.food_name,
        "calories": float(row.calories) if row.calories is not None else None,
        "protein_g": float(row.protein_g) if row.protein_g is not None else None,
        "carbs_g": float(row.carbs_g) if row.carbs_g is not None else None,
        "fat_g": float(row.fat_g) if row.fat_g is not None else None,
        "fiber_g": float(row.fiber_g) if row.fiber_g is not None else None,
    }
