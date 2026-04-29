from __future__ import annotations

from enum import Enum


class Imbalance(str, Enum):
    HIGH_CALORIES = "HIGH_CALORIES"
    LOW_PROTEIN = "LOW_PROTEIN"
    HIGH_CARBS = "HIGH_CARBS"
    HIGH_FAT = "HIGH_FAT"


class HealthGoal(str, Enum):
    WEIGHT_LOSS = "weight_loss"
    MUSCLE_GAIN = "muscle_gain"
    BALANCE = "balance"
    SPORT_PERFORMANCE = "sport_performance"


RECOMMENDATIONS_MATRIX: dict[tuple[Imbalance, HealthGoal], str] = {
    (Imbalance.HIGH_CALORIES, HealthGoal.WEIGHT_LOSS): (
        "Repas trop riche pour ton objectif de perte de poids. Reduis les "
        "portions et privilegie des aliments rassasiants comme les legumes "
        "verts ou les proteines maigres."
    ),
    (Imbalance.HIGH_CALORIES, HealthGoal.MUSCLE_GAIN): (
        "Bonne densite calorique pour soutenir ta prise de masse. Verifie "
        "tout de meme que l'apport en proteines suit (1.6 a 2g/kg/jour)."
    ),
    (Imbalance.HIGH_CALORIES, HealthGoal.BALANCE): (
        "Repas tres calorique. Pour garder ta journee equilibree, allege "
        "le repas suivant et ajoute davantage de legumes au prochain plat."
    ),
    (Imbalance.HIGH_CALORIES, HealthGoal.SPORT_PERFORMANCE): (
        "Repas dense en calories : ideal en pre-entrainement, allege le "
        "repas qui suit la seance."
    ),
    (Imbalance.LOW_PROTEIN, HealthGoal.WEIGHT_LOSS): (
        "Apport en proteines insuffisant. Pour preserver ta masse maigre "
        "pendant ta perte de poids, ajoute du poulet, du poisson ou des "
        "oeufs au prochain repas."
    ),
    (Imbalance.LOW_PROTEIN, HealthGoal.MUSCLE_GAIN): (
        "Pour ta prise de masse, ajoute une portion de poulet, oeufs ou "
        "legumineuses au prochain repas (objectif : 2g/kg/jour)."
    ),
    (Imbalance.LOW_PROTEIN, HealthGoal.BALANCE): (
        "Pense a renforcer l'apport en proteines au prochain repas avec du "
        "poisson, des legumineuses ou des produits laitiers."
    ),
    (Imbalance.LOW_PROTEIN, HealthGoal.SPORT_PERFORMANCE): (
        "Pas assez de proteines pour soutenir ta performance sportive. Vise "
        "1.6 a 2g/kg/jour en repartissant sur 3 a 4 repas (poulet, oeufs, whey)."
    ),
    (Imbalance.HIGH_CARBS, HealthGoal.WEIGHT_LOSS): (
        "Pour ton objectif de perte de poids, remplace les sucres rapides "
        "par des legumes verts ou des cereales completes."
    ),
    (Imbalance.HIGH_CARBS, HealthGoal.MUSCLE_GAIN): (
        "Bonne base de glucides pour ta prise de masse. Verifie que les "
        "proteines suivent (1.6 a 2g/kg/jour) et hydrate-toi correctement."
    ),
    (Imbalance.HIGH_CARBS, HealthGoal.BALANCE): (
        "Repas riche en glucides. Pour un meilleur equilibre, ajoute des "
        "legumes verts et privilegie les cereales completes au prochain repas."
    ),
    (Imbalance.HIGH_CARBS, HealthGoal.SPORT_PERFORMANCE): (
        "Apport eleve en glucides : parfait pour recharger tes reserves "
        "avant ou apres l'entrainement. Pense a accompagner d'une source "
        "de proteines."
    ),
    (Imbalance.HIGH_FAT, HealthGoal.WEIGHT_LOSS): (
        "Repas riche en graisses, peu compatible avec ta perte de poids. "
        "Reduis les fritures et les sauces, privilegie la cuisson vapeur "
        "ou au four."
    ),
    (Imbalance.HIGH_FAT, HealthGoal.MUSCLE_GAIN): (
        "Densite calorique elevee utile pour ta prise de masse, mais "
        "surveille la qualite des graisses : favorise les bonnes (avocat, "
        "huile d'olive, poissons gras)."
    ),
    (Imbalance.HIGH_FAT, HealthGoal.BALANCE): (
        "Repas trop gras pour rester equilibre. Allege le repas suivant et "
        "privilegie des cuissons plus simples (vapeur, four, grille)."
    ),
    (Imbalance.HIGH_FAT, HealthGoal.SPORT_PERFORMANCE): (
        "Repas riche en graisses : evite-le juste avant l'entrainement "
        "(digestion lente). Garde-le pour les repas eloignes des seances."
    ),
}

GENERIC_FALLBACK = (
    "Pense a equilibrer ton prochain repas pour rester aligne avec tes "
    "objectifs nutritionnels."
)


def get_recommendation(
    imbalance: Imbalance, health_goal: HealthGoal | None
) -> str:
    if health_goal is None:
        return GENERIC_FALLBACK
    return RECOMMENDATIONS_MATRIX.get(
        (imbalance, health_goal), GENERIC_FALLBACK
    )
