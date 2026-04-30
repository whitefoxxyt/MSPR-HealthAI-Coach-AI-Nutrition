from __future__ import annotations

from app.models.schemas import ServingSize, ServingSizeLabel

# Grammages PNNS (Programme National Nutrition Sante) issus des reperes
# mangerbouger.fr et ANSES (PNNS 4, table CIQUAL). 3 tailles indicatives par
# categorie : small (portion modeste), medium (portion de reference adulte),
# large (portion genereuse / sportif).
_PNNS_SERVING_SIZES: dict[str, list[ServingSize]] = {
    "legumes": [
        ServingSize(label=ServingSizeLabel.small, grams=80, description="80 g cuit"),
        ServingSize(label=ServingSizeLabel.medium, grams=200, description="200 g cuit"),
        ServingSize(label=ServingSizeLabel.large, grams=300, description="300 g cuit"),
    ],
    "feculents": [
        ServingSize(label=ServingSizeLabel.small, grams=100, description="100 g cuit"),
        ServingSize(label=ServingSizeLabel.medium, grams=200, description="200 g cuit"),
        ServingSize(label=ServingSizeLabel.large, grams=300, description="300 g cuit"),
    ],
    "viandes": [
        ServingSize(label=ServingSizeLabel.small, grams=70, description="70 g cuit"),
        ServingSize(label=ServingSizeLabel.medium, grams=100, description="100 g cuit"),
        ServingSize(label=ServingSizeLabel.large, grams=150, description="150 g cuit"),
    ],
    "poissons": [
        ServingSize(label=ServingSizeLabel.small, grams=70, description="70 g cuit"),
        ServingSize(label=ServingSizeLabel.medium, grams=100, description="100 g cuit"),
        ServingSize(label=ServingSizeLabel.large, grams=150, description="150 g cuit"),
    ],
    "fromages": [
        ServingSize(label=ServingSizeLabel.small, grams=20, description="20 g"),
        ServingSize(
            label=ServingSizeLabel.medium, grams=30, description="30 g (une portion)"
        ),
        ServingSize(label=ServingSizeLabel.large, grams=40, description="40 g"),
    ],
    "fruits": [
        ServingSize(label=ServingSizeLabel.small, grams=80, description="80 g cru"),
        ServingSize(
            label=ServingSizeLabel.medium, grams=150, description="150 g cru (un fruit)"
        ),
        ServingSize(label=ServingSizeLabel.large, grams=200, description="200 g cru"),
    ],
    "oleagineux": [
        ServingSize(
            label=ServingSizeLabel.small, grams=15, description="15 g (une poignee)"
        ),
        ServingSize(label=ServingSizeLabel.medium, grams=30, description="30 g"),
        ServingSize(label=ServingSizeLabel.large, grams=50, description="50 g"),
    ],
    "plats_composes": [
        ServingSize(label=ServingSizeLabel.small, grams=200, description="200 g"),
        ServingSize(label=ServingSizeLabel.medium, grams=350, description="350 g"),
        ServingSize(label=ServingSizeLabel.large, grams=500, description="500 g"),
    ],
    "desserts": [
        ServingSize(label=ServingSizeLabel.small, grams=50, description="50 g"),
        ServingSize(label=ServingSizeLabel.medium, grams=100, description="100 g"),
        ServingSize(label=ServingSizeLabel.large, grams=150, description="150 g"),
    ],
    "boissons": [
        ServingSize(label=ServingSizeLabel.small, grams=150, description="150 ml"),
        ServingSize(
            label=ServingSizeLabel.medium, grams=250, description="250 ml (un verre)"
        ),
        ServingSize(label=ServingSizeLabel.large, grams=350, description="350 ml"),
    ],
}

# Mapping HITL des 101 labels Food-101 vers les 10 categories PNNS.
# Pre-genere puis revu manuellement ligne par ligne (cf. issue NUT-49).
# Toute modification requiert une nouvelle validation humaine et la mise a
# jour du snapshot dans tests/unit/test_portion_sizes.py.
_FOOD101_TO_PNNS_CATEGORY: dict[str, str] = {
    "apple_pie": "desserts",
    "baby_back_ribs": "viandes",
    "baklava": "desserts",
    "beef_carpaccio": "viandes",
    "beef_tartare": "viandes",
    "beet_salad": "legumes",
    "beignets": "desserts",
    "bibimbap": "plats_composes",
    "bread_pudding": "desserts",
    "breakfast_burrito": "plats_composes",
    "bruschetta": "plats_composes",
    "caesar_salad": "legumes",
    "cannoli": "desserts",
    "caprese_salad": "legumes",
    "carrot_cake": "desserts",
    "ceviche": "poissons",
    "cheese_plate": "fromages",
    "cheesecake": "desserts",
    "chicken_curry": "plats_composes",
    "chicken_quesadilla": "plats_composes",
    "chicken_wings": "viandes",
    "chocolate_cake": "desserts",
    "chocolate_mousse": "desserts",
    "churros": "desserts",
    "clam_chowder": "plats_composes",
    "club_sandwich": "plats_composes",
    "crab_cakes": "poissons",
    "creme_brulee": "desserts",
    "croque_madame": "plats_composes",
    "cup_cakes": "desserts",
    "deviled_eggs": "plats_composes",
    "donuts": "desserts",
    "dumplings": "plats_composes",
    "edamame": "legumes",
    "eggs_benedict": "plats_composes",
    "escargots": "plats_composes",
    "falafel": "plats_composes",
    "filet_mignon": "viandes",
    "fish_and_chips": "plats_composes",
    "foie_gras": "viandes",
    "french_fries": "feculents",
    "french_onion_soup": "plats_composes",
    "french_toast": "plats_composes",
    "fried_calamari": "poissons",
    "fried_rice": "feculents",
    "frozen_yogurt": "desserts",
    "garlic_bread": "feculents",
    "gnocchi": "feculents",
    "greek_salad": "legumes",
    "grilled_cheese_sandwich": "plats_composes",
    "grilled_salmon": "poissons",
    "guacamole": "legumes",
    "gyoza": "plats_composes",
    "hamburger": "plats_composes",
    "hot_and_sour_soup": "plats_composes",
    "hot_dog": "plats_composes",
    "huevos_rancheros": "plats_composes",
    "hummus": "legumes",
    "ice_cream": "desserts",
    "lasagna": "plats_composes",
    "lobster_bisque": "plats_composes",
    "lobster_roll_sandwich": "plats_composes",
    "macaroni_and_cheese": "plats_composes",
    "macarons": "desserts",
    "miso_soup": "plats_composes",
    "mussels": "poissons",
    "nachos": "plats_composes",
    "omelette": "plats_composes",
    "onion_rings": "legumes",
    "oysters": "poissons",
    "pad_thai": "plats_composes",
    "paella": "plats_composes",
    "pancakes": "desserts",
    "panna_cotta": "desserts",
    "peking_duck": "viandes",
    "pho": "plats_composes",
    "pizza": "plats_composes",
    "pork_chop": "viandes",
    "poutine": "plats_composes",
    "prime_rib": "viandes",
    "pulled_pork_sandwich": "plats_composes",
    "ramen": "plats_composes",
    "ravioli": "plats_composes",
    "red_velvet_cake": "desserts",
    "risotto": "feculents",
    "samosa": "plats_composes",
    "sashimi": "poissons",
    "scallops": "poissons",
    "seaweed_salad": "legumes",
    "shrimp_and_grits": "plats_composes",
    "spaghetti_bolognese": "plats_composes",
    "spaghetti_carbonara": "plats_composes",
    "spring_rolls": "plats_composes",
    "steak": "viandes",
    "strawberry_shortcake": "desserts",
    "sushi": "plats_composes",
    "tacos": "plats_composes",
    "takoyaki": "plats_composes",
    "tiramisu": "desserts",
    "tuna_tartare": "poissons",
    "waffles": "desserts",
}


def get_serving_sizes(food_label: str) -> list[ServingSize]:
    category = _FOOD101_TO_PNNS_CATEGORY.get(food_label)
    if category is None:
        return [
            ServingSize(label=ServingSizeLabel.medium, grams=100, description="100 g")
        ]
    return list(_PNNS_SERVING_SIZES[category])
