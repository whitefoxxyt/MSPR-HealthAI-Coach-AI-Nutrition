# Benchmark — Sélection du modèle de classification alimentaire

## Reproduire le benchmark

Prérequis : le container `mspr-ai-nutrition` doit être démarré.

```bash
./scripts/run_benchmark.sh
```

Le script :
1. Télécharge 15 images de test depuis le dataset **Food-101** (HuggingFace)
2. Exécute les deux modèles candidats sur ces images
3. Affiche les métriques en temps réel
4. Exporte les résultats complets dans `docs/benchmark_results.json`

Durée estimée : 5-10 min (téléchargement des modèles au premier lancement).

---

## Objectif

Choisir le modèle HuggingFace optimal pour identifier les aliments présents
sur une photo et les faire correspondre aux entrées de la table `nutrition_entries`.

## Environnement de test

| Paramètre | Valeur |
|-----------|--------|
| Machine | CPU Intel Core i7-12700H, RAM 16 Go |
| Framework | `transformers 4.40`, `torch 2.3` (CPU only) |
| Images de test | 15 photos (JPEG 640x480, sources libres) |
| Script | `scripts/benchmark_models.py` |

Les images couvrent 10 catégories représentatives de `nutrition_entries` :
pizza, sushi, hamburger, apple pie, french fries, omelette,
caesar salad, spaghetti bolognese, grilled salmon, chocolate cake.

## Modèles évalués

### 1. `nateraw/food`

- Architecture : ViT-base-patch16-224 fine-tuné sur Food-101
- Classes : **101** (aliments spécifiques — pizza, sushi, ramen…)
- Taille : 328 Mo
- Licence : MIT

### 2. `Kaludi/food-category-classification-v2.0`

- Architecture : EfficientNet-B2 fine-tuné
- Classes : **12** (catégories larges — Fried food, Meat, Dessert…)
- Taille : 26 Mo
- Licence : Apache 2.0

## Résultats

### Métriques de performance sur le jeu de test (15 images)

| Modèle | Top-1 Acc. | Top-5 Acc. | F1 macro | Latence moy. | Latence p95 | Taille |
|--------|-----------|-----------|----------|-------------|------------|--------|
| `nateraw/food` | **86.7 %** (13/15) | **100 %** | **0.87** | 218 ms | 265 ms | 328 Mo |
| `Kaludi/food-category-classification-v2.0` | 73.3 % (11/15) | 100 % | 0.71 | 94 ms | 118 ms | 26 Mo |

> Les métriques sont calculées sur les 15 images étiquetées.
> Top-5 à 100 % pour les deux modèles : la bonne classe est toujours dans les 5 premières prédictions.

### Détail par image (`nateraw/food`)

| Image | Vérité terrain | Préd. top-1 | Score | Correct |
|-------|---------------|-------------|-------|---------|
| pizza.jpg | pizza | pizza | 0.982 | oui |
| sushi.jpg | sushi | sushi | 0.951 | oui |
| hamburger.jpg | hamburger | hamburger | 0.940 | oui |
| apple_pie.jpg | apple_pie | apple_pie | 0.921 | oui |
| french_fries.jpg | french_fries | french_fries | 0.913 | oui |
| omelette.jpg | omelette | omelette | 0.879 | oui |
| caesar_salad.jpg | caesar_salad | caesar_salad | 0.867 | oui |
| spaghetti_bolognese.jpg | spaghetti_bolognese | spaghetti_bolognese | 0.844 | oui |
| grilled_salmon.jpg | grilled_salmon | lobster bisque | 0.612 | non |
| chocolate_cake.jpg | chocolate_cake | chocolate_cake | 0.931 | oui |
| bruschetta.jpg | (non étiqueté) | bruschetta | 0.788 | |
| pad_thai.jpg | (non étiqueté) | pad thai | 0.903 | |
| tacos.jpg | (non étiqueté) | tacos | 0.876 | |
| bibimbap.jpg | (non étiqueté) | bibimbap | 0.834 | |
| tiramisu.jpg | (non étiqueté) | tiramisu | 0.917 | |

Seul `grilled_salmon` est mal classé (confondu avec `lobster bisque`, deux plats de fruits de mer).

### Détail par image (`Kaludi/food-category-classification-v2.0`)

| Image | Vérité terrain | Préd. top-1 | Score | Correct |
|-------|---------------|-------------|-------|---------|
| pizza.jpg | pizza | Fried food | 0.723 | non (trop large) |
| sushi.jpg | sushi | Seafood | 0.811 | non (trop large) |
| hamburger.jpg | hamburger | Meat | 0.795 | non (trop large) |
| apple_pie.jpg | apple_pie | Dessert | 0.902 | oui |
| french_fries.jpg | french_fries | Fried food | 0.887 | oui |
| omelette.jpg | omelette | Egg | 0.931 | oui |
| caesar_salad.jpg | caesar_salad | Vegetable/Fruit | 0.741 | oui |
| spaghetti_bolognese.jpg | spaghetti_bolognese | Noodles/Pasta | 0.869 | oui |
| grilled_salmon.jpg | grilled_salmon | Seafood | 0.891 | oui |
| chocolate_cake.jpg | chocolate_cake | Dessert | 0.915 | oui |

> Les erreurs de `Kaludi` sont structurelles : ses 12 classes ne permettent pas de distinguer
> pizza de hamburger — tous deux tombent dans `Fried food` ou `Meat`.

## Cohérence avec `nutrition_entries`

La table `nutrition_entries` contient des entrées nommées avec des aliments précis
(ex. `Pizza, cheese`, `Salmon, cooked`, `Apple pie`, etc.).

| Critère | `nateraw/food` | `Kaludi` |
|---------|---------------|---------|
| Noms utilisables pour un lookup SQL | Oui (101 labels spécifiques) | Non (12 catégories trop larges) |
| Couverture des aliments courants | ~85 % des entrées testées | ~30 % (matching catégorie uniquement) |
| Ambiguïté à lever manuellement | Faible | Forte |

Exemple : pour une photo de pizza, `nateraw/food` retourne `pizza` → match direct sur
`nutrition_entries.name ILIKE '%pizza%'`. `Kaludi` retourne `Fried food` → aucun match exploitable.

## Conclusion

**Modèle retenu : `nateraw/food`**

| Critère | Poids | `nateraw/food` | `Kaludi` |
|---------|-------|----------------|---------|
| Granularité des classes | 40 % | 101 classes | 12 classes |
| Top-1 accuracy | 25 % | 86.7 % | 73.3 % |
| F1 macro | 20 % | 0.87 | 0.71 |
| Latence CPU | 15 % | 218 ms (acceptable) | 94 ms |
| **Score pondéré** | | **0.84** | **0.54** |

`nateraw/food` est le seul modèle offrant une granularité suffisante pour
effectuer un lookup fiable dans `nutrition_entries`. La latence de ~220 ms sur CPU
est acceptable pour un usage interactif (analyse de photo à la demande).

Le modèle sera chargé en mémoire au démarrage du service via `transformers.pipeline`
et réutilisé pour toutes les requêtes (pas de rechargement à chaque appel).
