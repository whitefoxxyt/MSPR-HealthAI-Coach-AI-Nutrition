from __future__ import annotations

import re

import pytest

from app.models.schemas import FallbackMealPlan
from app.services.fallback_loader import load_fallback_plan

HEALTH_GOALS = ["weight_loss", "muscle_gain", "balance", "sport_performance"]
DIET_TYPES = ["omnivore", "vegetarien", "vegan", "sans_gluten"]
MATRIX = [(g, d) for g in HEALTH_GOALS for d in DIET_TYPES]

# Regle PRD : nombre de repas quotidiens par objectif sante.
MEALS_PER_DAY = {
    "weight_loss": 3,
    "balance": 3,
    "sport_performance": 4,
    "muscle_gain": 5,
}

# Listes d'ingredients incompatibles par regime. On compare via "in" sur l'ingredient
# en minuscule. Les "skip markers" desamorcent les faux positifs (lait d'amande,
# farine de sarrasin, etc.).
GLUTEN_KEYWORDS = [
    "ble", "pain", "pates", "spaghetti", "couscous", "semoule", "boulgour",
    "orge", "seigle", "epeautre", "tortilla", "wrap", "bagel", "biscotte",
    "croissant", "brioche", "muesli", "farine de ble", "farine d'epeautre",
    "farine de seigle", "farine complete", "lasagnes", "pate brisee",
    "pate a pizza", "pain burger",
]
GLUTEN_SKIP_MARKERS = [
    "sarrasin", "sans gluten", "de mais", "de riz", "de quinoa", "d'amande",
    "de coco", "de pois chiche",
]

MEAT_KEYWORDS = [
    "poulet", "boeuf", "porc", "veau", "agneau", "dinde", "canard", "lapin",
    "jambon", "bacon", "saucisse", "chorizo", "lardons", "steak", "escalope",
    "viande", "saumon", "thon", "cabillaud", "merlu", "sardine", "anchois",
    "crevette", "moule", "huitre", "poisson", "fruits de mer",
]

DAIRY_KEYWORDS = [
    "lait", "fromage", "yaourt", "beurre", "creme", "ricotta", "feta",
    "mozzarella", "parmesan", "comte", "emmental", "gruyere", "chevre",
    "brebis", "kefir", "skyr", "cottage cheese", "bechamel",
]
DAIRY_SKIP_MARKERS = [
    "d'amande", "de soja", " soja", "d'avoine", "de coco", "de riz",
    "de cacahuete", "vegetal", "vegetale", "vegan",
]

EGG_KEYWORDS = ["oeuf", "oeufs"]


def _contains_any(
    ingredient: str, keywords: list[str], skip_markers: list[str] = ()
) -> str | None:
    """Cherche un mot interdit en respectant les frontieres de mot.

    Les skip_markers desamorcent les faux positifs ("lait d'amande" / "lait
    de coco" ne comptent pas comme produits laitiers).
    """
    norm = ingredient.lower()
    if any(m in norm for m in skip_markers):
        return None
    for kw in keywords:
        # \b ne fonctionne pas autour des apostrophes, on utilise des classes explicites.
        pattern = r"(?:^|[^a-z])" + re.escape(kw) + r"(?:[^a-z]|$)"
        if re.search(pattern, norm):
            return kw
    return None


def test_known_combo_returns_dict_validating_schema():
    plan = load_fallback_plan("balance", "omnivore")

    assert plan is not None
    assert plan["fallback"] is True
    FallbackMealPlan.model_validate(plan)


def test_unknown_combo_returns_none():
    assert load_fallback_plan("balance", "carnivore") is None
    assert load_fallback_plan("supersize", "omnivore") is None


@pytest.mark.parametrize(("health_goal", "diet_type"), MATRIX)
def test_matrix_returns_valid_plan(health_goal: str, diet_type: str):
    plan = load_fallback_plan(health_goal, diet_type)

    assert plan is not None, f"plan manquant pour {health_goal}_{diet_type}"
    FallbackMealPlan.model_validate(plan)
    assert plan["fallback"] is True


@pytest.mark.parametrize(("health_goal", "diet_type"), MATRIX)
def test_seven_days_with_correct_meal_count(health_goal: str, diet_type: str):
    plan = load_fallback_plan(health_goal, diet_type)
    assert plan is not None

    days = plan["days"]
    assert len(days) == 7, f"{health_goal}_{diet_type} doit couvrir 7 jours, vu {len(days)}"

    expected_meals = MEALS_PER_DAY[health_goal]
    for day in days:
        assert len(day["meals"]) == expected_meals, (
            f"{health_goal}_{diet_type} jour {day['day']} : "
            f"{len(day['meals'])} repas, attendu {expected_meals}"
        )

    day_numbers = [d["day"] for d in days]
    assert day_numbers == list(range(1, 8))


def _all_ingredients(plan: dict) -> list[str]:
    return [ing for day in plan["days"] for meal in day["meals"] for ing in meal["ingredients"]]


@pytest.mark.parametrize("health_goal", HEALTH_GOALS)
def test_sans_gluten_excludes_gluten(health_goal: str):
    plan = load_fallback_plan(health_goal, "sans_gluten")
    assert plan is not None

    for ing in _all_ingredients(plan):
        hit = _contains_any(ing, GLUTEN_KEYWORDS, skip_markers=GLUTEN_SKIP_MARKERS)
        assert hit is None, f"sans_gluten contient gluten via '{ing}' (mot : {hit})"


@pytest.mark.parametrize("health_goal", HEALTH_GOALS)
def test_vegetarien_excludes_meat_and_fish(health_goal: str):
    plan = load_fallback_plan(health_goal, "vegetarien")
    assert plan is not None

    for ing in _all_ingredients(plan):
        hit = _contains_any(ing, MEAT_KEYWORDS)
        assert hit is None, f"vegetarien contient viande/poisson via '{ing}' (mot : {hit})"


@pytest.mark.parametrize("health_goal", HEALTH_GOALS)
def test_vegan_excludes_animal_products(health_goal: str):
    plan = load_fallback_plan(health_goal, "vegan")
    assert plan is not None

    for ing in _all_ingredients(plan):
        meat_hit = _contains_any(ing, MEAT_KEYWORDS)
        assert meat_hit is None, f"vegan contient viande/poisson via '{ing}' (mot : {meat_hit})"
        dairy_hit = _contains_any(ing, DAIRY_KEYWORDS, skip_markers=DAIRY_SKIP_MARKERS)
        assert dairy_hit is None, f"vegan contient produit laitier via '{ing}' (mot : {dairy_hit})"
        egg_hit = _contains_any(ing, EGG_KEYWORDS)
        assert egg_hit is None, f"vegan contient oeuf via '{ing}' (mot : {egg_hit})"
